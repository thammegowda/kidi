import DOMPurify from './libs/dompurify/purify.es.mjs';

let renderer;

function rendererMain() {
    mermaid.initialize({startOnLoad: false, securityLevel: 'strict', theme: 'neutral',
        maxTextSize: 20000, maxEdges: 200, suppressErrorRendering: true, htmlLabels: false,
        flowchart: {htmlLabels: false}, fontFamily: 'sans-serif'});
    let queue = Promise.resolve();
    addEventListener('message', event => {
        if (event.source !== parent || event.data?.type !== 'render') return;
        const {id, source} = event.data;
        queue = queue.then(async () => {
            try {
                const {svg} = await mermaid.render(`diagram${id}`, source);
                parent.postMessage({id, svg}, '*');
            } catch {
                parent.postMessage({id, error: true}, '*');
            }
        });
    });
    parent.postMessage({type: 'renderer-ready'}, '*');
}

function waitForMessage(frame, accepts, send) {
    return new Promise((resolve, reject) => {
        const cleanup = () => { clearTimeout(timer); removeEventListener('message', receive); };
        const receive = event => {
            if (event.source !== frame.contentWindow || !accepts(event.data)) return;
            cleanup();
            resolve(event.data);
        };
        const timer = setTimeout(() => { cleanup(); reject(new Error('Diagram rendering timed out')); }, 15000);
        addEventListener('message', receive);
        send();
    });
}

async function getRenderer() {
    if (!renderer) renderer = (async () => {
        const response = await fetch(new URL('./libs/mermaid/mermaid.min.js', import.meta.url));
        if (!response.ok) throw new Error('Diagram renderer unavailable');
        const source = await response.text();
        const frame = document.createElement('iframe');
        frame.sandbox = 'allow-scripts';
        frame.title = 'Diagram renderer';
        frame.setAttribute('aria-hidden', 'true');
        frame.tabIndex = -1;
        frame.style.cssText = 'position:fixed;left:-10000px;width:1000px;height:800px;visibility:hidden;border:0';
        const nonce = crypto.randomUUID();
        const script = `${source}\n;(${rendererMain.toString()})();`.replace(/<\/script/gi, '<\\/script');
        try {
            await waitForMessage(frame, data => data?.type === 'renderer-ready', () => {
                frame.srcdoc = `<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-${nonce}'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"><script nonce="${nonce}">${script}</script>`;
                document.body.append(frame);
            });
            return frame;
        } catch (error) { frame.remove(); throw error; }
    })().catch(error => { renderer = null; throw error; });
    return renderer;
}

export async function renderDiagram(source) {
    if (source.length > 20000) throw new Error('Diagram is too large to preview');
    const rendererFrame = await getRenderer();
    const id = crypto.randomUUID().replaceAll('-', '');
    const result = await waitForMessage(rendererFrame, data => data?.id === id,
        () => rendererFrame.contentWindow.postMessage({type: 'render', id, source}, '*'));
    if (result.error || typeof result.svg !== 'string') throw new Error('Diagram could not be rendered');
    const svg = DOMPurify.sanitize(result.svg, {USE_PROFILES: {svg: true, svgFilters: true},
        FORBID_TAGS: ['foreignObject', 'a']});
    const documentSvg = new DOMParser().parseFromString(svg, 'image/svg+xml').documentElement;
    const viewBox = documentSvg.getAttribute('viewBox')?.trim().split(/[\s,]+/).map(Number);
    const frame = document.createElement('iframe');
    frame.className = 'mermaid-diagram';
    frame.title = 'Mermaid diagram';
    frame.sandbox = '';
    frame.referrerPolicy = 'no-referrer';
    if (viewBox?.length === 4 && viewBox.every(Number.isFinite) && viewBox[2] > 0 && viewBox[3] > 0)
        frame.style.aspectRatio = `${viewBox[2]} / ${viewBox[3]}`;
    frame.srcdoc = `<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"><style>body{margin:0;background:white}svg{display:block;width:100%;height:auto;max-width:100%!important}</style>${svg}`;
    return frame;
}