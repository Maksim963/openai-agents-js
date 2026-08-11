import { describe, expect, it, vi } from 'vitest';
import { RealtimeAgent } from '../../src/realtimeAgent';
import { RealtimeSession } from '../../src/realtimeSession';
import {
  IncompleteRealtimeScenarioError,
  RealtimeCallMatcherError,
  RealtimeTransportNotClosedError,
  ScriptedRealtimeTransport,
  UnexpectedRealtimeCallError,
} from '../../src/testing';

describe('ScriptedRealtimeTransport', () => {
  it('drives a session with ordered calls and typed events', async () => {
    const transport = new ScriptedRealtimeTransport();
    const outputDeltas: string[] = [];
    transport.on('output_text_delta', (event) => {
      outputDeltas.push(event.delta);
    });
    const scenario = transport.runScenario(async ({ expectCall, emit }) => {
      await expectCall('connect', (call) => {
        expect(call.options.apiKeyProvided).toBe(true);
        expect(call.options).not.toHaveProperty('apiKey');
        expect(call.options.initialSessionConfig).toBeDefined();
      });
      await expectCall('sendMessage', (call) => {
        expect(call.message).toBe('Hello');
        expect(call.otherEventData).toEqual({ source: 'test' });
      });
      emit('turn_started', {
        type: 'response_started',
        providerData: { response: { id: 'response_1' } },
      });
      emit('output_text_delta', {
        type: 'output_text_delta',
        responseId: 'response_1',
        itemId: 'item_1',
        delta: 'Hi',
      });
      await expectCall('close');
    });
    const session = new RealtimeSession(
      new RealtimeAgent({ name: 'Assistant' }),
      { transport },
    );

    await session.connect({ apiKey: 'test' });
    session.sendMessage('Hello', { source: 'test' });
    session.close();
    await scenario;

    expect(outputDeltas).toEqual(['Hi']);
    expect(transport.calls.map((call) => call.method)).toEqual([
      'connect',
      'sendMessage',
      'close',
    ]);
    expect(() => transport.assertComplete()).not.toThrow();
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('matches calls that occurred before the expectation was registered', async () => {
    const transport = new ScriptedRealtimeTransport();
    transport.sendEvent({ type: 'session.update', session: {} });

    const call = await transport.expectCall('sendEvent');

    expect(call).toMatchObject({
      index: 1,
      event: { type: 'session.update' },
    });
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it('rejects an unexpected call without a time-based wait', async () => {
    const transport = new ScriptedRealtimeTransport();
    const scenario = transport.runScenario(async ({ expectCall }) => {
      await expectCall('sendMessage');
    });

    expect(() =>
      transport.sendAudio(new ArrayBuffer(1), { commit: true }),
    ).toThrow(UnexpectedRealtimeCallError);
    await expect(scenario).rejects.toBeInstanceOf(UnexpectedRealtimeCallError);
  });

  it('fails when a call matcher returns false', async () => {
    const transport = new ScriptedRealtimeTransport();
    const scenario = transport.runScenario(async ({ expectCall }) => {
      await expectCall('mute', () => false);
    });

    expect(() => transport.mute(true)).toThrow(RealtimeCallMatcherError);
    await expect(scenario).rejects.toBeInstanceOf(RealtimeCallMatcherError);
  });

  it('injects a connect failure and restores disconnected state', async () => {
    const transport = new ScriptedRealtimeTransport();
    const error = new Error('connect failed');
    transport.failNextCall('connect', error);
    const expectedCall = transport.expectCall('connect');

    await expect(transport.connect({ apiKey: 'test' })).rejects.toBe(error);
    await expect(expectedCall).resolves.toMatchObject({ method: 'connect' });
    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertComplete()).not.toThrow();
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('reserves a connect failure before a reentrant matcher call', async () => {
    const transport = new ScriptedRealtimeTransport();
    const injectedFailure = new Error('outer connect failed');
    let nestedConnect: Promise<void> | undefined;
    transport.failNextCall('connect', injectedFailure);
    const outerCall = transport.expectCall('connect', () => {
      nestedConnect = transport.connect({ apiKey: 'nested' });
    });

    await expect(transport.connect({ apiKey: 'outer' })).rejects.toBe(
      injectedFailure,
    );
    await outerCall;
    await expect(nestedConnect).resolves.toBeUndefined();
    await expect(transport.expectCall('connect')).resolves.toMatchObject({
      index: 2,
      method: 'connect',
    });
    expect(transport.status).toBe('connected');
    transport.disconnect();
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it('reserves a close failure before a reentrant matcher call', async () => {
    const transport = new ScriptedRealtimeTransport();
    await transport.connect({ apiKey: 'test' });
    await transport.expectCall('connect');
    const injectedFailure = new Error('outer close failed');
    transport.failNextCall('close', injectedFailure);
    const outerCall = transport.expectCall('close', () => {
      transport.close();
    });

    expect(() => transport.close()).toThrow(injectedFailure);
    await outerCall;
    await expect(transport.expectCall('close')).resolves.toMatchObject({
      index: 3,
      method: 'close',
    });
    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it.each(['connect matcher', 'connecting notification'] as const)(
    'restores disconnected state when a failed close supersedes an in-flight connection from the %s',
    async (trigger) => {
      const transport = new ScriptedRealtimeTransport();
      const injectedFailure = new Error('close failed');
      const statuses: string[] = [];
      let closeFailure: unknown;
      transport.failNextCall('close', injectedFailure);
      transport.on('connection_change', (status) => {
        statuses.push(status);
        if (trigger === 'connecting notification' && status === 'connecting') {
          try {
            transport.close();
          } catch (error) {
            closeFailure = error;
          }
        }
      });
      const connectCall = transport.expectCall('connect', () => {
        if (trigger === 'connect matcher') {
          try {
            transport.close();
          } catch (error) {
            closeFailure = error;
          }
        }
      });

      await expect(transport.connect({ apiKey: 'test' })).rejects.toMatchObject(
        {
          name: 'ScriptedRealtimeConnectionSupersededError',
        },
      );
      await connectCall;
      await expect(transport.expectCall('close')).resolves.toMatchObject({
        method: 'close',
      });

      expect(closeFailure).toBe(injectedFailure);
      expect(transport.status).toBe('disconnected');
      expect(statuses.at(-1)).toBe('disconnected');
      expect(() => transport.assertComplete()).not.toThrow();
      expect(() => transport.assertClosed()).not.toThrow();
    },
  );

  it('restores disconnected state when a nested failed close supersedes an outer close', async () => {
    const transport = new ScriptedRealtimeTransport();
    await transport.connect({ apiKey: 'test' });
    await transport.expectCall('connect');
    const injectedFailure = new Error('nested close failed');
    let closeFailure: unknown;
    const outerCloseCall = transport.expectCall('close', () => {
      transport.failNextCall('close', injectedFailure);
      try {
        transport.close();
      } catch (error) {
        closeFailure = error;
      }
    });

    transport.close();
    await outerCloseCall;
    await expect(transport.expectCall('close')).resolves.toMatchObject({
      index: 3,
      method: 'close',
    });

    expect(closeFailure).toBe(injectedFailure);
    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertComplete()).not.toThrow();
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('reserves a synchronous failure before a reentrant matcher call', async () => {
    const transport = new ScriptedRealtimeTransport();
    const injectedFailure = new Error('outer mute failed');
    transport.failNextCall('mute', injectedFailure);
    const outerCall = transport.expectCall('mute', () => {
      transport.mute(false);
    });

    expect(() => transport.mute(true)).toThrow(injectedFailure);
    await outerCall;
    await expect(transport.expectCall('mute')).resolves.toMatchObject({
      index: 2,
      method: 'mute',
      muted: false,
    });
    expect(transport.muted).toBe(false);
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it('preserves connect failures while completing state cleanup', async () => {
    const observerFailure = new Error('connecting observer failed');
    const observerTransport = new ScriptedRealtimeTransport();
    observerTransport.on('connection_change', (status) => {
      if (status === 'connecting') {
        throw observerFailure;
      }
    });

    await expect(observerTransport.connect({ apiKey: 'test' })).rejects.toBe(
      observerFailure,
    );
    expect(observerTransport.status).toBe('disconnected');

    const injectedFailure = new Error('injected connect failure');
    const cleanupFailure = new Error('cleanup observer failed');
    const injectedTransport = new ScriptedRealtimeTransport();
    injectedTransport.failNextCall('connect', injectedFailure);
    const connectCall = injectedTransport.expectCall('connect');
    injectedTransport.on('connection_change', (status) => {
      if (status === 'disconnected') {
        throw cleanupFailure;
      }
    });

    await expect(injectedTransport.connect({ apiKey: 'test' })).rejects.toBe(
      injectedFailure,
    );
    await connectCall;
    expect(injectedTransport.status).toBe('disconnected');
    expect(() => injectedTransport.assertComplete()).not.toThrow();
  });

  it('settles connect expectations before observer failures', async () => {
    const observerFailure = new Error('connecting observer failed');
    const injectedFailure = new Error('injected failure');
    const transport = new ScriptedRealtimeTransport();
    const connectCall = transport.expectCall('connect');
    transport.failNextCall('connect', injectedFailure);
    transport.on('connection_change', (status) => {
      if (status === 'connecting') {
        throw observerFailure;
      }
    });

    await expect(transport.connect({ apiKey: 'test' })).rejects.toBe(
      injectedFailure,
    );
    await expect(connectCall).resolves.toMatchObject({ method: 'connect' });
    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it.each(['close', 'disconnect'] as const)(
    'preserves a reentrant %s during the connecting notification',
    async (operation) => {
      const transport = new ScriptedRealtimeTransport();
      const scenario = transport.runScenario(async ({ expectCall }) => {
        await expectCall('connect');
        if (operation === 'close') {
          await expectCall('close');
        }
      });
      transport.on('connection_change', (status) => {
        if (status !== 'connecting') {
          return;
        }
        if (operation === 'close') {
          transport.close();
        } else {
          transport.disconnect();
        }
      });

      await expect(transport.connect({ apiKey: 'test' })).rejects.toMatchObject(
        { name: 'ScriptedRealtimeConnectionSupersededError' },
      );
      await scenario;
      expect(transport.status).toBe('disconnected');
      expect(transport.calls.map((call) => call.method)).toEqual(
        operation === 'close' ? ['connect', 'close'] : ['connect'],
      );
      expect(() => transport.assertComplete()).not.toThrow();
      expect(() => transport.assertClosed()).not.toThrow();
    },
  );

  it.each(['close', 'disconnect'] as const)(
    'preserves a reentrant %s from a connect matcher',
    async (operation) => {
      const transport = new ScriptedRealtimeTransport();
      const connectCall = transport.expectCall('connect', () => {
        if (operation === 'close') {
          transport.close();
        } else {
          transport.disconnect();
        }
      });

      await expect(transport.connect({ apiKey: 'test' })).rejects.toMatchObject(
        { name: 'ScriptedRealtimeConnectionSupersededError' },
      );
      await connectCall;
      if (operation === 'close') {
        await expect(transport.expectCall('close')).resolves.toMatchObject({
          method: 'close',
        });
      }
      expect(transport.status).toBe('disconnected');
      expect(transport.calls.map((call) => call.method)).toEqual(
        operation === 'close' ? ['connect', 'close'] : ['connect'],
      );
      expect(() => transport.assertComplete()).not.toThrow();
      expect(() => transport.assertClosed()).not.toThrow();
    },
  );

  it('emits a controlled disconnect and transport error', async () => {
    const transport = new ScriptedRealtimeTransport();
    const statuses: string[] = [];
    const errors: unknown[] = [];
    transport.on('connection_change', (status) => statuses.push(status));
    transport.on('error', (event) => errors.push(event.error));
    const connectCall = transport.expectCall('connect');
    await transport.connect({ apiKey: 'test' });
    await connectCall;

    const error = new Error('disconnected');
    transport.disconnect(error);

    expect(statuses).toEqual(['connecting', 'connected', 'disconnected']);
    expect(errors).toEqual([error]);
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('closes idempotently and records only the effective close', async () => {
    const transport = new ScriptedRealtimeTransport();
    const scenario = transport.runScenario(async ({ expectCall }) => {
      await expectCall('connect');
      await expectCall('close');
    });

    await transport.connect({ apiKey: 'test' });
    transport.close();
    transport.close();
    await scenario;

    expect(
      transport.calls.filter((call) => call.method === 'close'),
    ).toHaveLength(1);
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it('keeps terminal state when close observers throw', async () => {
    const transport = new ScriptedRealtimeTransport();
    const connectCall = transport.expectCall('connect');
    await transport.connect({ apiKey: 'test' });
    await connectCall;
    const closeCall = transport.expectCall('close');
    const observerFailure = new Error('close observer failed');
    transport.on('connection_change', (status) => {
      if (status === 'disconnected') {
        throw observerFailure;
      }
    });

    expect(() => transport.close()).toThrow(observerFailure);
    await closeCall;
    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('preserves a reconnect started by a close matcher', async () => {
    const transport = new ScriptedRealtimeTransport();
    await transport.connect({ apiKey: 'test' });
    await transport.expectCall('connect');
    let reconnect: Promise<void> | undefined;
    const closeCall = transport.expectCall('close', () => {
      reconnect = transport.connect({ apiKey: 'test' });
    });
    const reconnectCall = transport.expectCall('connect');

    transport.close();
    await closeCall;
    await reconnectCall;
    await reconnect;

    expect(transport.status).toBe('connected');
    expect(transport.calls.map((call) => call.method)).toEqual([
      'connect',
      'close',
      'connect',
    ]);
    expect(() => transport.assertComplete()).not.toThrow();
  });

  it('preserves a matcher disconnect when close fails', async () => {
    const transport = new ScriptedRealtimeTransport();
    await transport.connect({ apiKey: 'test' });
    await transport.expectCall('connect');
    const injectedFailure = new Error('close failed');
    transport.failNextCall('close', injectedFailure);
    const closeCall = transport.expectCall('close', () => {
      transport.disconnect();
    });

    expect(() => transport.close()).toThrow(injectedFailure);
    await closeCall;

    expect(transport.status).toBe('disconnected');
    expect(() => transport.assertComplete()).not.toThrow();
    expect(() => transport.assertClosed()).not.toThrow();
  });

  it('attempts error notification after a disconnect observer throws', async () => {
    const transport = new ScriptedRealtimeTransport();
    const observerFailure = new Error('disconnect observer failed');
    const transportFailure = new Error('transport failed');
    const errors: unknown[] = [];
    const connectCall = transport.expectCall('connect');
    await transport.connect({ apiKey: 'test' });
    await connectCall;
    transport.on('connection_change', (status) => {
      if (status === 'disconnected') {
        throw observerFailure;
      }
    });
    transport.on('error', (event) => errors.push(event.error));

    expect(() => transport.disconnect(transportFailure)).toThrow(
      observerFailure,
    );
    expect(errors).toEqual([transportFailure]);
    expect(transport.status).toBe('disconnected');
  });

  it('reports incomplete scenarios and open transports', async () => {
    const transport = new ScriptedRealtimeTransport();
    transport.sendEvent({ type: 'session.update' });

    expect(() => transport.assertComplete()).toThrow(
      IncompleteRealtimeScenarioError,
    );

    const connected = new ScriptedRealtimeTransport();
    const connectCall = connected.expectCall('connect');
    await connected.connect({ apiKey: 'test' });
    await connectCall;
    expect(() => connected.assertClosed()).toThrow(
      RealtimeTransportNotClosedError,
    );
  });

  it('supports assertions in matchers', async () => {
    const transport = new ScriptedRealtimeTransport();
    const matcher = vi.fn((call: { muted: boolean }) => {
      expect(call.muted).toBe(true);
    });
    const expectedCall = transport.expectCall('mute', matcher);

    transport.mute(true);
    await expectedCall;

    expect(matcher).toHaveBeenCalledOnce();
  });
});
