import type { AgentInputItem } from '../types';
import { UserError } from '../errors';
import type { SessionHistoryMutation } from './session';

const OPTIONAL_FUNCTION_CALL_FIELDS = new Set([
  'id',
  'namespace',
  'status',
  'providerData',
]);

/**
 * Applies persisted-history mutations and returns a new canonical item list.
 */
export function applySessionHistoryMutations(
  items: AgentInputItem[],
  mutations: SessionHistoryMutation[],
): AgentInputItem[] {
  if (mutations.length === 0) {
    return items.map(cloneSessionHistoryValueSafely);
  }
  const normalizedMutations = snapshotSessionHistoryMutations(mutations);
  if (normalizedMutations.some((mutation) => mutation.expected !== undefined)) {
    assertSessionHistoryValuesSafe([
      ...items,
      ...normalizedMutations.flatMap((mutation) =>
        mutation.expected === undefined
          ? [mutation.replacement]
          : [mutation.expected, mutation.replacement],
      ),
    ]);
  }

  let nextItems = items;

  for (const mutation of normalizedMutations) {
    if (mutation.type === 'replace_function_call') {
      nextItems = applyReplaceFunctionCallMutation(nextItems, mutation);
    }
  }

  return nextItems;
}

function applyReplaceFunctionCallMutation(
  items: AgentInputItem[],
  mutation: Extract<SessionHistoryMutation, { type: 'replace_function_call' }>,
): AgentInputItem[] {
  const replacement = cloneSessionHistoryValueSafely(mutation.replacement);
  if (mutation.expected === undefined) {
    return applyLegacyReplaceFunctionCallMutation(
      items,
      mutation.callId,
      replacement,
    );
  }

  let matchingIndex = -1;
  let replacementAlreadyApplied = false;
  try {
    for (let index = items.length - 1; index >= 0; index -= 1) {
      const item = items[index];
      if (item?.type !== 'function_call' || item.callId !== mutation.callId) {
        continue;
      }
      if (sessionHistoryItemsMatch(item, mutation.expected)) {
        matchingIndex = index;
        break;
      }
      if (sessionHistoryItemsMatch(item, replacement)) {
        replacementAlreadyApplied = true;
        break;
      }
    }
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }

  if (replacementAlreadyApplied) {
    return items.map(cloneSessionHistoryValueSafely);
  }
  if (matchingIndex < 0) {
    throw new UserError(
      `Session history mutation could not find the expected function call for call ID ${mutation.callId}.`,
    );
  }

  return items.map((item, index) =>
    index === matchingIndex
      ? replacement
      : cloneSessionHistoryValueSafely(item),
  );
}

function applyLegacyReplaceFunctionCallMutation(
  items: AgentInputItem[],
  callId: string,
  replacement: Extract<AgentInputItem, { type: 'function_call' }>,
): AgentInputItem[] {
  const nextItems: AgentInputItem[] = [];
  let keptReplacement = false;

  for (const item of items) {
    if (item.type === 'function_call' && item.callId === callId) {
      if (!keptReplacement) {
        nextItems.push(replacement);
        keptReplacement = true;
      }
      continue;
    }
    nextItems.push(cloneSessionHistoryValueSafely(item));
  }

  return nextItems;
}

export function sessionHistoryItemsMatch(
  actual: AgentInputItem,
  expected: AgentInputItem,
): boolean {
  return sessionHistoryValuesMatch(actual, expected);
}

export function sessionHistoryValuesMatch(
  actual: unknown,
  expected: unknown,
): boolean {
  try {
    return (
      JSON.stringify(sortSessionHistoryValue(actual)) ===
      JSON.stringify(sortSessionHistoryValue(expected))
    );
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }
}

export function snapshotSessionHistoryValue<T>(value: T): T {
  try {
    return sortSessionHistoryValue(value) as T;
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }
}

function assertSessionHistoryValuesSafe(values: unknown[]): void {
  try {
    for (const value of values) {
      sortSessionHistoryValue(value);
    }
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }
}

function cloneSessionHistoryValueSafely<T>(value: T): T {
  try {
    return structuredClone(value);
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }
}

export function snapshotSessionHistoryMutations(
  mutations: SessionHistoryMutation[],
): SessionHistoryMutation[] {
  try {
    return mutations.map((mutation) => {
      const descriptors = Object.getOwnPropertyDescriptors(mutation);
      for (const key of Reflect.ownKeys(descriptors)) {
        const descriptor = Reflect.get(descriptors, key) as PropertyDescriptor;
        if (descriptor.enumerable && !('value' in descriptor)) {
          throw new TypeError(
            'Accessor-backed session history mutation is unsupported.',
          );
        }
      }

      const type = readMutationDataProperty(descriptors, 'type');
      const callId = readMutationDataProperty(descriptors, 'callId');
      const replacement = readMutationDataProperty(descriptors, 'replacement');
      const expectedDescriptor = descriptors.expected;
      if (
        type !== 'replace_function_call' ||
        typeof callId !== 'string' ||
        replacement === undefined ||
        (expectedDescriptor !== undefined && !('value' in expectedDescriptor))
      ) {
        throw new TypeError('Invalid session history mutation.');
      }

      const expected = expectedDescriptor?.value;
      if (expected === undefined) {
        return {
          type,
          callId,
          replacement: replacement as Extract<
            AgentInputItem,
            { type: 'function_call' }
          >,
        };
      }

      const expectedSnapshot = sortSessionHistoryValue(expected);
      const replacementSnapshot = sortSessionHistoryValue(replacement);
      if (
        !isFunctionCallSnapshot(expectedSnapshot) ||
        !isFunctionCallSnapshot(replacementSnapshot)
      ) {
        throw new TypeError('Invalid expected-bearing history mutation.');
      }
      return {
        type,
        callId,
        expected: expectedSnapshot,
        replacement: replacementSnapshot,
      };
    });
  } catch {
    throw new UserError('Session history items could not be compared safely.');
  }
}

function readMutationDataProperty(
  descriptors: PropertyDescriptorMap,
  key: string,
): unknown {
  const descriptor = descriptors[key];
  if (!descriptor || !('value' in descriptor)) {
    throw new TypeError('Invalid session history mutation.');
  }
  return descriptor.value;
}

function sortSessionHistoryValue(
  value: unknown,
  ancestors: WeakSet<object> = new WeakSet(),
): unknown {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'boolean'
  ) {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || Object.is(value, -0)) {
      throw new TypeError('Lossy session history number.');
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new TypeError('Unsupported session history value.');
  }
  if (ancestors.has(value)) {
    throw new TypeError('Cyclic session history value.');
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      const descriptors = Object.getOwnPropertyDescriptors(value);
      const result: unknown[] = new Array(value.length);
      for (const key of Reflect.ownKeys(descriptors)) {
        const descriptor = Reflect.get(descriptors, key) as PropertyDescriptor;
        if (key === 'length' || !descriptor.enumerable) {
          continue;
        }
        if (
          typeof key !== 'string' ||
          !isCanonicalArrayIndex(key, value.length)
        ) {
          throw new TypeError(
            'Enumerable array properties are unsupported in session history.',
          );
        }
      }
      for (let index = 0; index < value.length; index += 1) {
        const descriptor = descriptors[index];
        if (!descriptor || !descriptor.enumerable || !('value' in descriptor)) {
          throw new TypeError(
            'Sparse or accessor-backed session history arrays are unsupported.',
          );
        }
        result[index] = sortSessionHistoryValue(descriptor.value, ancestors);
      }
      return result;
    }
    if (!isPlainObject(value)) {
      throw new TypeError('Unsupported session history object.');
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    const typeDescriptor = descriptors.type;
    const isFunctionCall =
      typeDescriptor !== undefined &&
      'value' in typeDescriptor &&
      typeDescriptor.value === 'function_call';
    const entries: Array<[string, unknown]> = [];
    for (const key of Reflect.ownKeys(descriptors)) {
      const descriptor = Reflect.get(descriptors, key) as PropertyDescriptor;
      if (!descriptor.enumerable) {
        continue;
      }
      if (typeof key !== 'string') {
        throw new TypeError(
          'Enumerable symbol properties are unsupported in session history.',
        );
      }
      if (!('value' in descriptor)) {
        throw new TypeError('Accessor-backed session history is unsupported.');
      }
      if (
        isFunctionCall &&
        OPTIONAL_FUNCTION_CALL_FIELDS.has(key) &&
        descriptor.value === undefined
      ) {
        continue;
      }
      entries.push([key, sortSessionHistoryValue(descriptor.value, ancestors)]);
    }
    entries.sort(([left], [right]) =>
      left < right ? -1 : left > right ? 1 : 0,
    );
    return Object.fromEntries(entries);
  } finally {
    ancestors.delete(value);
  }
}

function isCanonicalArrayIndex(key: string, length: number): boolean {
  const index = Number(key);
  return (
    Number.isInteger(index) &&
    index >= 0 &&
    index < length &&
    String(index) === key
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isFunctionCallSnapshot(
  value: unknown,
): value is Extract<AgentInputItem, { type: 'function_call' }> {
  return (
    isPlainObject(value) &&
    value.type === 'function_call' &&
    typeof value.callId === 'string' &&
    typeof value.name === 'string' &&
    typeof value.arguments === 'string'
  );
}
