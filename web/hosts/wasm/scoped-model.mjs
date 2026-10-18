/// The loader's /model namespace is private to each model session; all sessions share one Wasm heap.
export function scopedModel(module, directory) {
    if (!/^\/models\/[a-z0-9-]+$/.test(directory)) throw new Error('Invalid model session directory');
    const allocations = new Map();
    const path = value => typeof value === 'string' && (value === '/model' || value.startsWith('/model/'))
        ? directory + value.slice('/model'.length) : value;
    module.FS.mkdirTree('/models');
    const fs = new Proxy(module.FS, {
        get(target, name) {
            if (typeof target[name] !== 'function') return target[name];
            return (...args) => {
                if (args.length) args[0] = path(args[0]);
                if (name === 'rename' || name === 'symlink') args[1] = path(args[1]);
                try { return target[name](...args); }
                catch (error) {
                    throw new Error(`Model filesystem ${String(name)}(${String(args[0])}): ${error.message || `errno ${error.errno}`}`,
                        {cause: error});
                }
            };
        },
    });
    const owner = Object.create(null);
    Object.defineProperties(owner, {
        HEAPU8: {get: () => module.HEAPU8},
        allocatedBytes: {get: () => [...allocations.values()].reduce((total, bytes) => total + bytes, 0)},
        FS: {value: fs},
        _malloc: {value: size => {
            const pointer = Number(module._malloc(size));
            if (pointer) allocations.set(pointer, Number(size));
            return pointer;
        }},
        _free: {value: pointer => {
            allocations.delete(pointer);
            module._free(pointer);
        }},
        dispose: {value: () => {
            const remove = directory => {
                for (const name of module.FS.readdir(directory)) {
                    if (name === '.' || name === '..') continue;
                    const child = `${directory}/${name}`;
                    if (module.FS.isDir(module.FS.stat(child).mode)) remove(child);
                    else module.FS.unlink(child);
                }
                module.FS.rmdir(directory);
            };
            if (module.FS.analyzePath(directory).exists) remove(directory);
            for (const pointer of allocations.keys()) module._free(pointer);
            allocations.clear();
            owner.kidiExternal = null;
        }},
    });
    return owner;
}
