import test from 'node:test';
import assert from 'node:assert/strict';
import {awaitWithSignal, RuntimeScheduler} from '../../web/hosts/wasm/runtime-scheduler.mjs';

test('GPU regions serialize and speech priority applies between regions', async () => {
    const scheduler = new RuntimeScheduler();
    const calls = [];
    let release;
    const blocked = scheduler.run(1, () => new Promise(resolve => { release = resolve; calls.push('active'); }));
    await new Promise(resolve => queueMicrotask(resolve));
    const background = scheduler.run(0, async () => calls.push('background'));
    const foreground = scheduler.run(2, async () => calls.push('speech'));
    release();
    await Promise.all([blocked, background, foreground]);
    assert.deepEqual(calls, ['active', 'speech', 'background']);
});

test('aborted queued regions never run', async () => {
    const scheduler = new RuntimeScheduler();
    const controller = new AbortController();
    let ran = false;
    const request = scheduler.run(1, () => { ran = true; }, controller.signal);
    controller.abort();
    await assert.rejects(request, error => error.name === 'AbortError');
    assert.equal(ran, false);
});

test('a cancelled logical client stops waiting without aborting another client shared load', async () => {
    const controller = new AbortController();
    let complete;
    const shared = new Promise(resolve => { complete = resolve; });
    const cancelled = awaitWithSignal(shared, controller.signal);
    controller.abort();
    await assert.rejects(cancelled, error => error.name === 'AbortError');
    complete('ready');
    assert.equal(await awaitWithSignal(shared), 'ready');
    assert.throws(() => awaitWithSignal(shared, controller.signal), error => error.name === 'AbortError');
});
