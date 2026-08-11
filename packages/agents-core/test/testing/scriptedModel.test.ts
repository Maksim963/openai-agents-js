import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { Agent, run, tool, Usage } from '../../src';
import type { ModelRequest } from '../../src/model';
import type { StreamEvent } from '../../src/types/protocol';
import {
  InvalidScriptedModelStepError,
  ScriptedModel,
  UnexpectedModelCallError,
  UnconsumedModelStepsError,
  assistantMessage,
  functionCall,
  modelError,
  modelResponder,
  modelResponse,
  modelStream,
  modelStreamResponder,
} from '../../src/testing';

describe('ScriptedModel', () => {
  it('drives a multi-turn tool workflow and records calls', async () => {
    const weather = tool({
      name: 'get_weather',
      description: 'Gets the weather.',
      parameters: z.object({ city: z.string() }),
      execute: async ({ city }) => `${city}: sunny`,
    });
    const model = new ScriptedModel([
      modelResponse([
        functionCall('get_weather', { city: 'Tokyo' }, { callId: 'call_1' }),
      ]),
      modelResponse([assistantMessage('It is sunny.')]),
    ]);
    const agent = new Agent({
      name: 'Weather assistant',
      model,
      tools: [weather],
    });

    const result = await run(agent, 'Weather in Tokyo?');

    expect(result.finalOutput).toBe('It is sunny.');
    expect(model.calls).toHaveLength(2);
    expect(model.firstCall).toMatchObject({
      index: 1,
      streamed: false,
      request: {
        input: [
          {
            type: 'message',
            role: 'user',
            content: 'Weather in Tokyo?',
          },
        ],
      },
    });
    expect(model.lastCall?.request.input).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'function_call_result' }),
      ]),
    );
    expect(model.remainingSteps).toBe(0);
    expect(() => model.assertComplete()).not.toThrow();
  });

  it('derives a response from the recorded call', async () => {
    const model = new ScriptedModel([
      modelResponder(async (call) => {
        await Promise.resolve();
        return [
          assistantMessage(`call ${call.index}: ${String(call.request.input)}`),
        ];
      }),
    ]);

    const response = await model.getResponse(makeRequest('hello'));

    expect(response.output).toEqual([assistantMessage('call 1: hello')]);
    expect(response.responseId).toBe('scripted-response-1');
  });

  it('preserves a complete non-streaming response object', async () => {
    const response = {
      output: [assistantMessage('hello')],
      usage: new Usage(),
    };
    const model = new ScriptedModel([modelResponse(response)]);

    await expect(model.getResponse(makeRequest('hello'))).resolves.toBe(
      response,
    );
  });

  it('converts a response into normalized streaming events', async () => {
    const response = {
      output: [assistantMessage('hello', { id: 'message_1' })],
      usage: new Usage({
        requests: 1,
        inputTokens: 2,
        outputTokens: 1,
        totalTokens: 3,
      }),
      responseId: 'response_1',
      requestId: 'request_1',
      providerData: { vendor: 'metadata' },
    };
    const directModel = new ScriptedModel([modelResponse(response)]);
    const streamingModel = new ScriptedModel([
      modelResponder(async () => response),
    ]);

    const directResponse = await directModel.getResponse(makeRequest('hello'));

    const events: StreamEvent[] = [];
    for await (const event of streamingModel.getStreamedResponse(
      makeRequest('hello'),
    )) {
      events.push(event);
    }

    expect(directResponse.providerData).toEqual({ vendor: 'metadata' });
    expect(events).toEqual([
      { type: 'response_started' },
      {
        type: 'output_text_delta',
        itemId: 'message_1',
        delta: 'hello',
        providerData: { annotations: [] },
      },
      expect.objectContaining({
        type: 'response_done',
        response: expect.objectContaining({
          id: 'response_1',
          requestId: 'request_1',
          usage: expect.objectContaining({ totalTokens: 3 }),
          providerData: { vendor: 'metadata' },
        }),
      }),
    ]);
    expect(streamingModel.lastCall?.streamed).toBe(true);
  });

  it('assigns streamed steps and records calls in invocation order', async () => {
    const model = new ScriptedModel([
      modelResponse([assistantMessage('first')]),
      modelResponse([assistantMessage('second')]),
    ]);
    const firstStream = model.getStreamedResponse(makeRequest('first call'));
    const secondStream = model.getStreamedResponse(makeRequest('second call'));

    expect(model.calls.map((call) => call.request.input)).toEqual([
      'first call',
      'second call',
    ]);
    expect(model.remainingSteps).toBe(0);

    const secondEvents = await collectEvents(secondStream);
    const firstEvents = await collectEvents(firstStream);

    expect(getStreamText(firstEvents)).toBe('first');
    expect(getStreamText(secondEvents)).toBe('second');
  });

  it('passes through a raw normalized stream', async () => {
    const events: StreamEvent[] = [
      { type: 'output_text_delta', delta: 'custom' },
      {
        type: 'response_done',
        response: {
          id: 'custom_response',
          usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
          output: [assistantMessage('custom')],
        },
      } as StreamEvent,
    ];
    const model = new ScriptedModel([modelStream(events)]);

    const received: StreamEvent[] = [];
    for await (const event of model.getStreamedResponse(makeRequest('hello'))) {
      received.push(event);
    }

    expect(received).toEqual(events);
  });

  it('derives a raw normalized stream from the recorded call', async () => {
    const model = new ScriptedModel([
      modelStreamResponder(async (call) => {
        await Promise.resolve();
        return (async function* () {
          yield {
            type: 'output_text_delta',
            delta: `${call.index}:${String(call.request.input)}`,
          } satisfies StreamEvent;
          yield {
            type: 'response_done',
            response: {
              id: 'dynamic-stream',
              usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
              output: [assistantMessage('done')],
            },
          } as StreamEvent;
        })();
      }),
    ]);

    const events = await collectEvents(
      model.getStreamedResponse(makeRequest('hello')),
    );

    expect(events[0]).toEqual({
      type: 'output_text_delta',
      delta: '1:hello',
    });
    expect(model.firstCall).toMatchObject({ index: 1, streamed: true });
  });

  it('propagates an error and supplies scripted retry advice', async () => {
    const error = new Error('temporary failure');
    const model = new ScriptedModel([
      modelError(error, ({ attempt }) => ({
        suggested: attempt === 1,
        replaySafety: 'safe',
      })),
    ]);
    const request = makeRequest('hello');

    await expect(model.getResponse(request)).rejects.toBe(error);
    await expect(
      model.getRetryAdvice({
        request,
        error,
        stream: false,
        attempt: 1,
      }),
    ).resolves.toEqual({ suggested: true, replaySafety: 'safe' });
  });

  it('keeps retry advice scoped to each reused error step', async () => {
    const error = new Error('shared failure');
    const nonStreamingRequest = makeRequest('non-streaming');
    const streamingRequest = makeRequest('streaming');
    const missingAdviceRequest = makeRequest('missing advice');
    const model = new ScriptedModel([
      modelError(error, { suggested: false }),
      modelError(error, { suggested: true, replaySafety: 'safe' }),
      modelError(error),
    ]);

    const nonStreamingFailure = model.getResponse(nonStreamingRequest);
    const streamingFailure = collectEvents(
      model.getStreamedResponse(streamingRequest),
    );
    const missingAdviceFailure = model.getResponse(missingAdviceRequest);
    await expect(nonStreamingFailure).rejects.toBe(error);
    await expect(streamingFailure).rejects.toBe(error);
    await expect(missingAdviceFailure).rejects.toBe(error);

    await expect(
      model.getRetryAdvice({
        request: streamingRequest,
        error,
        stream: true,
        attempt: 1,
      }),
    ).resolves.toEqual({ suggested: true, replaySafety: 'safe' });
    await expect(
      model.getRetryAdvice({
        request: nonStreamingRequest,
        error,
        stream: false,
        attempt: 1,
      }),
    ).resolves.toEqual({ suggested: false });
    await expect(
      model.getRetryAdvice({
        request: missingAdviceRequest,
        error,
        stream: false,
        attempt: 1,
      }),
    ).resolves.toBeUndefined();
  });

  it('prioritizes exact request identity for reused errors', async () => {
    const error = new Error('shared failure');
    const sharedRequest = makeRequest('shared input');
    const firstRequest = { ...sharedRequest };
    const secondRequest = { ...sharedRequest };
    const model = new ScriptedModel([
      modelError(error, { suggested: false }),
      modelError(error, { suggested: true }),
    ]);

    await Promise.allSettled([
      model.getResponse(firstRequest),
      model.getResponse(secondRequest),
    ]);

    await expect(
      model.getRetryAdvice({
        request: secondRequest,
        error,
        stream: false,
        attempt: 1,
      }),
    ).resolves.toEqual({ suggested: true });
    await expect(
      model.getRetryAdvice({
        request: firstRequest,
        error,
        stream: false,
        attempt: 1,
      }),
    ).resolves.toEqual({ suggested: false });
  });

  it('rejects raw stream steps in non-streaming mode', async () => {
    const model = new ScriptedModel([
      modelStream([]),
      modelStreamResponder(() => []),
    ]);

    await expect(model.getResponse(makeRequest('hello'))).rejects.toThrow(
      InvalidScriptedModelStepError,
    );
    await expect(model.getResponse(makeRequest('hello'))).rejects.toThrow(
      InvalidScriptedModelStepError,
    );
  });

  it('fails on unexpected calls and unconsumed steps', async () => {
    const empty = new ScriptedModel();
    await expect(empty.getResponse(makeRequest('hello'))).rejects.toThrow(
      UnexpectedModelCallError,
    );

    const pending = new ScriptedModel([
      modelResponse([assistantMessage('unused')]),
    ]);
    expect(() => pending.assertComplete()).toThrow(UnconsumedModelStepsError);
  });

  it('checks abort signals before consuming a step', async () => {
    const controller = new AbortController();
    controller.abort();
    const model = new ScriptedModel([
      modelResponse([assistantMessage('unused')]),
    ]);

    await expect(
      model.getResponse(makeRequest('hello', controller.signal)),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(model.remainingSteps).toBe(1);
    expect(model.calls).toHaveLength(0);
  });
});

function makeRequest(input: string, signal?: AbortSignal): ModelRequest {
  return {
    input,
    modelSettings: {},
    tools: [],
    outputType: 'text',
    handoffs: [],
    tracing: false,
    signal,
  };
}

async function collectEvents(
  events: AsyncIterable<StreamEvent>,
): Promise<StreamEvent[]> {
  const collected: StreamEvent[] = [];
  for await (const event of events) {
    collected.push(event);
  }
  return collected;
}

function getStreamText(events: StreamEvent[]): string {
  return events
    .filter((event) => event.type === 'output_text_delta')
    .map((event) => event.delta)
    .join('');
}
