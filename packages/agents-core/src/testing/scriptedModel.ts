import {
  Model,
  ModelRequest,
  ModelResponse,
  ModelRetryAdvice,
  ModelRetryAdviceRequest,
} from '../model';
import type {
  AssistantMessageItem,
  FunctionCallItem,
  StreamEvent,
} from '../types/protocol';
import { Usage } from '../usage';

export type ScriptedModelResponse = ModelResponse | ModelResponse['output'];

export type RecordedModelCall = Readonly<{
  /** The one-based position of this model call. */
  index: number;

  /** Whether the runner requested a streamed response. */
  streamed: boolean;

  /** A shallow, frozen snapshot of the model request. */
  request: Readonly<ModelRequest>;
}>;

export type ScriptedModelRetryAdvice =
  | ModelRetryAdvice
  | ((
      request: ModelRetryAdviceRequest,
    ) => ModelRetryAdvice | undefined | Promise<ModelRetryAdvice | undefined>);

export type ScriptedModelResponseStep = {
  type: 'response';
  response: ScriptedModelResponse;
};

export type ScriptedModelErrorStep = {
  type: 'error';
  error: unknown;
  retryAdvice?: ScriptedModelRetryAdvice;
};

export type ScriptedModelResponderStep = {
  type: 'responder';
  respond: (
    call: RecordedModelCall,
  ) => ScriptedModelResponse | Promise<ScriptedModelResponse>;
};

export type ScriptedModelStream =
  Iterable<StreamEvent> | AsyncIterable<StreamEvent>;

export type ScriptedModelStreamStep = {
  type: 'stream';
  events: ScriptedModelStream;
};

export type ScriptedModelStreamResponderStep = {
  type: 'stream_responder';
  respond: (
    call: RecordedModelCall,
  ) => ScriptedModelStream | Promise<ScriptedModelStream>;
};

export type ScriptedModelStep =
  | ScriptedModelResponseStep
  | ScriptedModelErrorStep
  | ScriptedModelResponderStep
  | ScriptedModelStreamStep
  | ScriptedModelStreamResponderStep;

export type AssistantMessageOptions = {
  id?: string;
  phase?: AssistantMessageItem['phase'];
  providerData?: Record<string, unknown>;
};

export type FunctionCallOptions = {
  callId: string;
  id?: string;
  namespace?: string;
  providerData?: Record<string, unknown>;
};

export class UnexpectedModelCallError extends Error {
  constructor(call: RecordedModelCall) {
    super(
      `ScriptedModel received unexpected ${call.streamed ? 'streaming' : 'non-streaming'} call #${call.index}; no scripted steps remain.`,
    );
    this.name = 'UnexpectedModelCallError';
  }
}

export class UnconsumedModelStepsError extends Error {
  constructor(remainingSteps: number) {
    super(
      `ScriptedModel has ${remainingSteps} unconsumed scripted step${remainingSteps === 1 ? '' : 's'}.`,
    );
    this.name = 'UnconsumedModelStepsError';
  }
}

export class InvalidScriptedModelStepError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidScriptedModelStepError';
  }
}

/** Creates a scripted normalized model response step. */
export function modelResponse(
  response: ScriptedModelResponse,
): ScriptedModelResponseStep {
  return { type: 'response', response };
}

/** Creates a scripted model failure step. */
export function modelError(
  error: unknown,
  retryAdvice?: ScriptedModelRetryAdvice,
): ScriptedModelErrorStep {
  return { type: 'error', error, retryAdvice };
}

/** Creates a response step that derives its output from the recorded call. */
export function modelResponder(
  respond: ScriptedModelResponderStep['respond'],
): ScriptedModelResponderStep {
  return { type: 'responder', respond };
}

/** Creates a raw normalized stream step. */
export function modelStream(
  events: ScriptedModelStreamStep['events'],
): ScriptedModelStreamStep {
  return { type: 'stream', events };
}

/** Creates a raw normalized stream derived from the recorded call. */
export function modelStreamResponder(
  respond: ScriptedModelStreamResponderStep['respond'],
): ScriptedModelStreamResponderStep {
  return { type: 'stream_responder', respond };
}

/** Creates a normalized assistant text message. */
export function assistantMessage(
  text: string,
  options: AssistantMessageOptions = {},
): AssistantMessageItem {
  return {
    type: 'message',
    role: 'assistant',
    status: 'completed',
    id: options.id,
    phase: options.phase,
    content: [
      {
        type: 'output_text',
        text,
        providerData: { annotations: [] },
      },
    ],
    providerData: options.providerData,
  };
}

/** Creates a normalized function call item. */
export function functionCall(
  name: string,
  args: string | Record<string, unknown>,
  options: FunctionCallOptions,
): FunctionCallItem {
  return {
    type: 'function_call',
    name,
    callId: options.callId,
    id: options.id ?? options.callId,
    namespace: options.namespace,
    status: 'completed',
    arguments: typeof args === 'string' ? args : JSON.stringify(args),
    providerData: options.providerData,
  };
}

/**
 * A deterministic model test double that consumes one scripted step per call.
 */
export class ScriptedModel implements Model {
  readonly #steps: ScriptedModelStep[];
  readonly #calls: RecordedModelCall[] = [];
  readonly #retryAdvice = new Map<
    unknown,
    Array<{
      request: ModelRequest;
      streamed: boolean;
      advice: ScriptedModelRetryAdvice | undefined;
    }>
  >();

  constructor(steps: Iterable<ScriptedModelStep> = []) {
    this.#steps = Array.from(steps);
  }

  /** All calls recorded by this model in invocation order. */
  get calls(): readonly RecordedModelCall[] {
    return this.#calls;
  }

  /** The first recorded call, if one has occurred. */
  get firstCall(): RecordedModelCall | undefined {
    return this.#calls[0];
  }

  /** The most recent recorded call, if one has occurred. */
  get lastCall(): RecordedModelCall | undefined {
    return this.#calls.at(-1);
  }

  /** The number of scripted steps that have not been consumed. */
  get remainingSteps(): number {
    return this.#steps.length;
  }

  /** Adds steps to the end of the script. */
  enqueue(...steps: ScriptedModelStep[]): void {
    this.#steps.push(...steps);
  }

  /** Throws when the script still contains unconsumed steps. */
  assertComplete(): void {
    if (this.#steps.length > 0) {
      throw new UnconsumedModelStepsError(this.#steps.length);
    }
  }

  async getResponse(request: ModelRequest): Promise<ModelResponse> {
    throwIfAborted(request.signal);
    const { call, step } = this.#recordAndTakeStep(request, false);

    if (step.type === 'stream' || step.type === 'stream_responder') {
      throw new InvalidScriptedModelStepError(
        `ScriptedModel cannot use raw stream step for non-streaming call #${call.index}.`,
      );
    }

    if (step.type === 'error') {
      this.#rememberRetryAdvice(step, request, false);
      throw step.error;
    }

    const response =
      step.type === 'responder' ? await step.respond(call) : step.response;
    throwIfAborted(request.signal);
    return normalizeResponse(response, call.index, false);
  }

  getStreamedResponse(request: ModelRequest): AsyncIterable<StreamEvent> {
    throwIfAborted(request.signal);
    const { call, step } = this.#recordAndTakeStep(request, true);

    return this.#processStreamedStep(request, call, step);
  }

  async *#processStreamedStep(
    request: ModelRequest,
    call: RecordedModelCall,
    step: ScriptedModelStep,
  ): AsyncIterable<StreamEvent> {
    if (step.type === 'error') {
      this.#rememberRetryAdvice(step, request, true);
      throw step.error;
    }

    if (step.type === 'stream' || step.type === 'stream_responder') {
      const events =
        step.type === 'stream' ? step.events : await step.respond(call);
      for await (const event of events) {
        throwIfAborted(request.signal);
        yield event;
      }
      throwIfAborted(request.signal);
      return;
    }

    const response = normalizeResponse(
      step.type === 'responder' ? await step.respond(call) : step.response,
      call.index,
      true,
    );
    yield* streamResponse(response, request.signal);
  }

  async getRetryAdvice(
    request: ModelRetryAdviceRequest,
  ): Promise<ModelRetryAdvice | undefined> {
    const records = this.#retryAdvice.get(request.error);
    if (!records) {
      return undefined;
    }

    let recordIndex = records.findIndex(
      (record) =>
        record.streamed === request.stream &&
        record.request === request.request,
    );
    if (recordIndex === -1) {
      recordIndex = records.findIndex(
        (record) =>
          record.streamed === request.stream &&
          requestsShareCallInputs(record.request, request.request),
      );
    }
    if (recordIndex === -1) {
      recordIndex = records.findIndex(
        (record) => record.streamed === request.stream,
      );
    }
    if (recordIndex === -1) {
      return undefined;
    }

    const [record] = records.splice(recordIndex, 1);
    if (records.length === 0) {
      this.#retryAdvice.delete(request.error);
    }
    const advice = record?.advice;
    if (typeof advice === 'function') {
      return await advice(request);
    }
    return advice;
  }

  #recordAndTakeStep(
    request: ModelRequest,
    streamed: boolean,
  ): { call: RecordedModelCall; step: ScriptedModelStep } {
    const call: RecordedModelCall = Object.freeze({
      index: this.#calls.length + 1,
      streamed,
      request: Object.freeze({ ...request }),
    });
    this.#calls.push(call);

    const step = this.#steps.shift();
    if (!step) {
      throw new UnexpectedModelCallError(call);
    }
    return { call, step };
  }

  #rememberRetryAdvice(
    step: ScriptedModelErrorStep,
    request: ModelRequest,
    streamed: boolean,
  ): void {
    const records = this.#retryAdvice.get(step.error) ?? [];
    records.push({ request, streamed, advice: step.retryAdvice });
    this.#retryAdvice.set(step.error, records);
  }
}

function requestsShareCallInputs(
  recorded: ModelRequest,
  advised: ModelRequest,
): boolean {
  return (
    recorded.input === advised.input &&
    recorded.modelSettings === advised.modelSettings &&
    recorded.tools === advised.tools &&
    recorded.handoffs === advised.handoffs
  );
}

function normalizeResponse(
  response: ScriptedModelResponse,
  callIndex: number,
  requireResponseId: boolean,
): ModelResponse {
  if (Array.isArray(response)) {
    return {
      output: response,
      usage: new Usage(),
      responseId: `scripted-response-${callIndex}`,
    };
  }

  if (!requireResponseId || response.responseId) {
    return response;
  }

  return {
    ...response,
    responseId: response.responseId ?? `scripted-response-${callIndex}`,
  };
}

async function* streamResponse(
  response: ModelResponse,
  signal?: AbortSignal,
): AsyncIterable<StreamEvent> {
  throwIfAborted(signal);
  yield { type: 'response_started' };

  for (const item of response.output) {
    throwIfAborted(signal);
    if (
      item.type !== 'message' ||
      item.role !== 'assistant' ||
      !Array.isArray(item.content)
    ) {
      continue;
    }
    for (const part of item.content) {
      if (part.type === 'output_text') {
        yield {
          type: 'output_text_delta',
          itemId: item.id,
          delta: part.text,
          providerData: part.providerData,
        };
      }
    }
  }

  throwIfAborted(signal);
  yield {
    type: 'response_done',
    response: {
      id: response.responseId!,
      requestId: response.requestId,
      usage: {
        requests: response.usage.requests,
        inputTokens: response.usage.inputTokens,
        outputTokens: response.usage.outputTokens,
        totalTokens: response.usage.totalTokens,
        inputTokensDetails:
          response.usage.inputTokensDetails.length > 0
            ? response.usage.inputTokensDetails
            : undefined,
        outputTokensDetails:
          response.usage.outputTokensDetails.length > 0
            ? response.usage.outputTokensDetails
            : undefined,
        requestUsageEntries: response.usage.requestUsageEntries?.map(
          (entry) => ({
            inputTokens: entry.inputTokens,
            outputTokens: entry.outputTokens,
            totalTokens: entry.totalTokens,
            inputTokensDetails: entry.inputTokensDetails,
            outputTokensDetails: entry.outputTokensDetails,
            endpoint: entry.endpoint,
          }),
        ),
      },
      rawUsage: response.rawUsage,
      providerData: response.providerData,
      output: response.output,
    },
  } as StreamEvent;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }
  const error = new Error('Scripted model request was aborted.');
  error.name = 'AbortError';
  throw error;
}
