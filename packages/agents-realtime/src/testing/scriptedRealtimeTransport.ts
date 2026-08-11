import { RuntimeEventEmitter } from '@openai/agents-core/_shims';
import { EventEmitterDelegate } from '@openai/agents-core/utils';
import type {
  RealtimeClientMessage,
  RealtimeSessionConfig,
  RealtimeUserInput,
} from '../clientMessages';
import type {
  RealtimeItem,
  RealtimeMcpCallApprovalRequestItem,
} from '../items';
import type {
  RealtimeTransportLayer,
  RealtimeTransportLayerConnectOptions,
} from '../transportLayer';
import type {
  RealtimeTransportEventTypes,
  TransportError,
  TransportToolCallEvent,
} from '../transportLayerEvents';

export type RealtimeTransportCall =
  | {
      method: 'connect';
      options: Omit<RealtimeTransportLayerConnectOptions, 'apiKey'> & {
        apiKeyProvided: boolean;
      };
    }
  | { method: 'sendEvent'; event: RealtimeClientMessage }
  | { method: 'requestResponse'; response?: Record<string, any> }
  | {
      method: 'sendMessage';
      message: RealtimeUserInput;
      otherEventData: Record<string, any>;
      options?: { triggerResponse?: boolean };
    }
  | {
      method: 'addImage';
      image: string;
      options?: { triggerResponse?: boolean };
    }
  | {
      method: 'sendAudio';
      audio: ArrayBuffer;
      options: { commit?: boolean };
    }
  | {
      method: 'updateSessionConfig';
      config: Partial<RealtimeSessionConfig>;
    }
  | { method: 'close' }
  | { method: 'mute'; muted: boolean }
  | {
      method: 'sendFunctionCallOutput';
      toolCall: TransportToolCallEvent;
      output: string;
      startResponse: boolean;
    }
  | { method: 'interrupt' }
  | {
      method: 'resetHistory';
      oldHistory: RealtimeItem[];
      newHistory: RealtimeItem[];
    }
  | {
      method: 'sendMcpResponse';
      approvalRequest: RealtimeMcpCallApprovalRequestItem;
      approved: boolean;
      reason?: string;
    };

export type RecordedRealtimeTransportCall = Readonly<
  RealtimeTransportCall & { index: number }
>;

export type RealtimeTransportMethod = RealtimeTransportCall['method'];

export type RealtimeTransportCallFor<TMethod extends RealtimeTransportMethod> =
  Extract<RecordedRealtimeTransportCall, { method: TMethod }>;

export type RealtimeCallMatcher<TMethod extends RealtimeTransportMethod> = (
  call: RealtimeTransportCallFor<TMethod>,
) => boolean | void;

export type ScriptedRealtimeExpectCall = <
  TMethod extends RealtimeTransportMethod,
>(
  method: TMethod,
  matcher?: RealtimeCallMatcher<TMethod>,
) => Promise<RealtimeTransportCallFor<TMethod>>;

export type ScriptedRealtimeEmit = <
  TEvent extends keyof RealtimeTransportEventTypes,
>(
  event: TEvent,
  ...args: RealtimeTransportEventTypes[TEvent]
) => boolean;

export type ScriptedRealtimeScenarioContext = {
  expectCall: ScriptedRealtimeExpectCall;
  emit: ScriptedRealtimeEmit;
};

type PendingExpectation = {
  method: RealtimeTransportMethod;
  matcher?: (call: RecordedRealtimeTransportCall) => boolean | void;
  resolve: (call: RecordedRealtimeTransportCall) => void;
  reject: (error: unknown) => void;
};

export class UnexpectedRealtimeCallError extends Error {
  constructor(
    expectedMethod: RealtimeTransportMethod,
    actualCall: RecordedRealtimeTransportCall,
  ) {
    super(
      `ScriptedRealtimeTransport expected ${expectedMethod} but received ${actualCall.method} call #${actualCall.index}.`,
    );
    this.name = 'UnexpectedRealtimeCallError';
  }
}

export class RealtimeCallMatcherError extends Error {
  constructor(call: RecordedRealtimeTransportCall) {
    super(
      `ScriptedRealtimeTransport matcher rejected ${call.method} call #${call.index}.`,
    );
    this.name = 'RealtimeCallMatcherError';
  }
}

export class IncompleteRealtimeScenarioError extends Error {
  constructor(
    unconsumedCalls: number,
    pendingExpectations: number,
    pendingFailures: number,
  ) {
    super(
      `ScriptedRealtimeTransport scenario is incomplete: ${unconsumedCalls} unconsumed call${unconsumedCalls === 1 ? '' : 's'}, ${pendingExpectations} pending expectation${pendingExpectations === 1 ? '' : 's'}, and ${pendingFailures} pending failure${pendingFailures === 1 ? '' : 's'}.`,
    );
    this.name = 'IncompleteRealtimeScenarioError';
  }
}

export class RealtimeTransportNotClosedError extends Error {
  constructor(status: ScriptedRealtimeTransport['status']) {
    super(
      `ScriptedRealtimeTransport is not closed; current status is ${status}.`,
    );
    this.name = 'RealtimeTransportNotClosedError';
  }
}

/**
 * An in-memory Realtime transport for deterministic session tests.
 */
export class ScriptedRealtimeTransport
  extends EventEmitterDelegate<RealtimeTransportEventTypes>
  implements RealtimeTransportLayer
{
  protected eventEmitter =
    new RuntimeEventEmitter<RealtimeTransportEventTypes>();

  status: 'connected' | 'disconnected' | 'connecting' | 'disconnecting' =
    'disconnected';
  muted: boolean | null = false;

  readonly #calls: RecordedRealtimeTransportCall[] = [];
  readonly #unconsumedCalls: RecordedRealtimeTransportCall[] = [];
  readonly #expectations: PendingExpectation[] = [];
  readonly #failures = new Map<RealtimeTransportMethod, unknown[]>();
  #lifecycleGeneration = 0;

  /** All outbound calls recorded in invocation order. */
  get calls(): readonly RecordedRealtimeTransportCall[] {
    return this.#calls;
  }

  /** Registers an ordered expectation for the next outbound call. */
  expectCall<TMethod extends RealtimeTransportMethod>(
    method: TMethod,
    matcher?: RealtimeCallMatcher<TMethod>,
  ): Promise<RealtimeTransportCallFor<TMethod>> {
    const existingCall = this.#unconsumedCalls.shift();
    if (existingCall) {
      try {
        this.#validateExpectation(method, matcher, existingCall);
        return Promise.resolve(
          existingCall as RealtimeTransportCallFor<TMethod>,
        );
      } catch (error) {
        return Promise.reject(error);
      }
    }

    return new Promise<RealtimeTransportCallFor<TMethod>>((resolve, reject) => {
      this.#expectations.push({
        method,
        matcher: matcher as PendingExpectation['matcher'],
        resolve: (call) => resolve(call as RealtimeTransportCallFor<TMethod>),
        reject,
      });
    });
  }

  /** Runs a scenario with bound call expectations and event emission. */
  async runScenario<TResult>(
    scenario: (
      context: ScriptedRealtimeScenarioContext,
    ) => TResult | Promise<TResult>,
  ): Promise<TResult> {
    return await scenario({
      expectCall: this.expectCall.bind(this) as ScriptedRealtimeExpectCall,
      emit: this.emit.bind(this) as ScriptedRealtimeEmit,
    });
  }

  /** Makes the next call to a transport method throw the supplied error. */
  failNextCall(method: RealtimeTransportMethod, error: unknown): void {
    const failures = this.#failures.get(method) ?? [];
    failures.push(error);
    this.#failures.set(method, failures);
  }

  /** Emits a controlled terminal disconnect and optional transport error. */
  disconnect(error?: unknown): void {
    this.#lifecycleGeneration += 1;
    let notificationError: unknown;
    if (this.status !== 'disconnected') {
      this.status = 'disconnected';
      try {
        this.emit('connection_change', 'disconnected');
      } catch (caughtError) {
        notificationError = caughtError;
      }
    }
    if (typeof error !== 'undefined') {
      const transportError: TransportError = { type: 'error', error };
      try {
        this.emit('error', transportError);
      } catch (caughtError) {
        notificationError ??= caughtError;
      }
    }
    if (typeof notificationError !== 'undefined') {
      throw notificationError;
    }
  }

  /** Throws when calls or expectations remain unmatched. */
  assertComplete(): void {
    const pendingFailures = Array.from(this.#failures.values()).reduce(
      (total, failures) => total + failures.length,
      0,
    );
    if (
      this.#unconsumedCalls.length > 0 ||
      this.#expectations.length > 0 ||
      pendingFailures > 0
    ) {
      throw new IncompleteRealtimeScenarioError(
        this.#unconsumedCalls.length,
        this.#expectations.length,
        pendingFailures,
      );
    }
  }

  /** Throws unless the transport is disconnected. */
  assertClosed(): void {
    if (this.status !== 'disconnected') {
      throw new RealtimeTransportNotClosedError(this.status);
    }
  }

  async connect(options: RealtimeTransportLayerConnectOptions): Promise<void> {
    const generation = ++this.#lifecycleGeneration;
    try {
      this.status = 'connecting';
      this.#recordAndThrowNextFailure({
        method: 'connect',
        options: {
          model: options.model,
          url: options.url,
          callId: options.callId,
          initialSessionConfig: options.initialSessionConfig,
          apiKeyProvided: typeof options.apiKey !== 'undefined',
        },
      });
      this.#assertCurrentConnectionAttempt(generation, 'connecting');
      const connectingNotificationFailure =
        this.#emitConnectionChange('connecting');
      this.#assertCurrentConnectionAttempt(generation, 'connecting');
      if (connectingNotificationFailure.present) {
        throw connectingNotificationFailure.error;
      }
      this.status = 'connected';
      const connectedNotificationFailure =
        this.#emitConnectionChange('connected');
      this.#assertCurrentConnectionAttempt(generation, 'connected');
      if (connectedNotificationFailure.present) {
        throw connectedNotificationFailure.error;
      }
    } catch (error) {
      if (this.#lifecycleGeneration === generation) {
        this.#lifecycleGeneration += 1;
        this.status = 'disconnected';
        try {
          this.emit('connection_change', 'disconnected');
        } catch {
          // Preserve the primary connection failure.
        }
      }
      throw error;
    }
  }

  sendEvent(event: RealtimeClientMessage): void {
    this.#recordAndThrowNextFailure({ method: 'sendEvent', event });
  }

  requestResponse(response?: Record<string, any>): void {
    this.#recordAndThrowNextFailure({ method: 'requestResponse', response });
  }

  sendMessage(
    message: RealtimeUserInput,
    otherEventData: Record<string, any>,
    options?: { triggerResponse?: boolean },
  ): void {
    this.#recordAndThrowNextFailure({
      method: 'sendMessage',
      message,
      otherEventData,
      options,
    });
  }

  addImage(image: string, options?: { triggerResponse?: boolean }): void {
    this.#recordAndThrowNextFailure({ method: 'addImage', image, options });
  }

  sendAudio(audio: ArrayBuffer, options: { commit?: boolean }): void {
    this.#recordAndThrowNextFailure({ method: 'sendAudio', audio, options });
  }

  updateSessionConfig(config: Partial<RealtimeSessionConfig>): void {
    this.#recordAndThrowNextFailure({ method: 'updateSessionConfig', config });
  }

  close(): void {
    if (this.status === 'disconnected') {
      return;
    }

    const generation = ++this.#lifecycleGeneration;
    const previousStatus = this.status;
    this.status = 'disconnecting';
    try {
      this.#recordAndThrowNextFailure({ method: 'close' });
      if (
        this.#lifecycleGeneration !== generation ||
        this.status !== 'disconnecting'
      ) {
        return;
      }
      this.status = 'disconnected';
      this.emit('connection_change', 'disconnected');
    } catch (error) {
      if (
        this.#lifecycleGeneration === generation &&
        this.status === 'disconnecting'
      ) {
        this.status =
          previousStatus === 'connected' ? 'connected' : 'disconnected';
        if (this.status === 'disconnected') {
          try {
            this.emit('connection_change', 'disconnected');
          } catch {
            // Preserve the primary close failure.
          }
        }
      }
      throw error;
    }
  }

  mute(muted: boolean): void {
    this.#recordAndThrowNextFailure({ method: 'mute', muted });
    this.muted = muted;
  }

  sendFunctionCallOutput(
    toolCall: TransportToolCallEvent,
    output: string,
    startResponse: boolean,
  ): void {
    this.#recordAndThrowNextFailure({
      method: 'sendFunctionCallOutput',
      toolCall,
      output,
      startResponse,
    });
  }

  interrupt(): void {
    this.#recordAndThrowNextFailure({ method: 'interrupt' });
  }

  resetHistory(oldHistory: RealtimeItem[], newHistory: RealtimeItem[]): void {
    this.#recordAndThrowNextFailure({
      method: 'resetHistory',
      oldHistory,
      newHistory,
    });
  }

  sendMcpResponse(
    approvalRequest: RealtimeMcpCallApprovalRequestItem,
    approved: boolean,
    reason?: string,
  ): void {
    this.#recordAndThrowNextFailure({
      method: 'sendMcpResponse',
      approvalRequest,
      approved,
      reason,
    });
  }

  #record(call: RealtimeTransportCall): void {
    const recordedCall = Object.freeze({
      ...call,
      index: this.#calls.length + 1,
    }) as RecordedRealtimeTransportCall;
    this.#calls.push(recordedCall);

    const expectation = this.#expectations.shift();
    if (!expectation) {
      this.#unconsumedCalls.push(recordedCall);
      return;
    }

    try {
      this.#validateExpectation(
        expectation.method,
        expectation.matcher,
        recordedCall,
      );
      expectation.resolve(recordedCall);
    } catch (error) {
      expectation.reject(error);
      throw error;
    }
  }

  #validateExpectation<TMethod extends RealtimeTransportMethod>(
    method: TMethod,
    matcher: RealtimeCallMatcher<TMethod> | undefined,
    call: RecordedRealtimeTransportCall,
  ): void {
    if (call.method !== method) {
      throw new UnexpectedRealtimeCallError(method, call);
    }
    if (
      matcher &&
      matcher(call as RealtimeTransportCallFor<TMethod>) === false
    ) {
      throw new RealtimeCallMatcherError(call);
    }
  }

  #recordAndThrowNextFailure(call: RealtimeTransportCall): void {
    const failure = this.#takeNextFailure(call.method);
    try {
      this.#record(call);
    } catch (error) {
      if (failure.present) {
        const failures = this.#failures.get(call.method) ?? [];
        failures.unshift(failure.error);
        this.#failures.set(call.method, failures);
      }
      throw error;
    }
    if (failure.present) {
      throw failure.error;
    }
  }

  #takeNextFailure(
    method: RealtimeTransportMethod,
  ): { present: false } | { present: true; error: unknown } {
    const failures = this.#failures.get(method);
    if (!failures || failures.length === 0) {
      return { present: false };
    }
    const error = failures.shift();
    if (failures.length === 0) {
      this.#failures.delete(method);
    }
    return { present: true, error };
  }

  #emitConnectionChange(
    status: 'connected' | 'connecting',
  ): { present: false } | { present: true; error: unknown } {
    try {
      this.emit('connection_change', status);
      return { present: false };
    } catch (error) {
      return { present: true, error };
    }
  }

  #assertCurrentConnectionAttempt(
    generation: number,
    expectedStatus: 'connecting' | 'connected',
  ): void {
    if (
      this.#lifecycleGeneration !== generation ||
      this.status !== expectedStatus
    ) {
      const error = new Error(
        'ScriptedRealtimeTransport connection attempt was superseded by another lifecycle operation.',
      );
      error.name = 'ScriptedRealtimeConnectionSupersededError';
      throw error;
    }
  }
}
