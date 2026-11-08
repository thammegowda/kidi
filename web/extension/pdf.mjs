const MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024;
const MAX_TEXT_BYTES = 64 * 1024;
const MAX_PAGES = 1000;

async function downloadPdf(url) {
  if (!['https:', 'http:'].includes(new URL(url).protocol))
    throw new Error('Only HTTP and HTTPS PDF documents can be captured.');
  const response = await fetch(url, {credentials: 'include', signal: AbortSignal.timeout(60000)});
  if (!response.ok) throw new Error(`Unable to download this PDF: HTTP ${response.status}.`);
  if (Number(response.headers.get('content-length')) > MAX_DOWNLOAD_BYTES) {
    await response.body?.cancel();
    throw new Error('This PDF exceeds the 32 MiB download limit.');
  }
  if (!response.body) throw new Error('The PDF download returned no data.');
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > MAX_DOWNLOAD_BYTES) throw new Error('This PDF exceeds the 32 MiB download limit.');
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  const data = new Uint8Array(await new Blob(chunks).arrayBuffer());
  if (!new TextDecoder().decode(data.subarray(0, 1024)).includes('%PDF-'))
    throw new Error('The current page did not return a PDF document. Open its direct PDF URL and try again.');
  return data;
}

export async function extractPdf(url) {
  const data = await downloadPdf(url);
  const {getDocument, PDFWorker} = await import('./libs/pdfjs/pdf.min.mjs');
  const port = new Worker(new URL('./libs/pdfjs/pdf.worker.min.mjs', import.meta.url), {type: 'module'});
  const worker = new PDFWorker({port});
  let loading;
  try {
    loading = getDocument({
      data, worker, useWasm: false, disableFontFace: true, useSystemFonts: true,
      cMapUrl: new URL('./libs/pdfjs/cmaps/', import.meta.url).href, cMapPacked: true,
      standardFontDataUrl: new URL('./libs/pdfjs/standard_fonts/', import.meta.url).href,
    });
    const document = await loading.promise;
    const {info, metadata} = await document.getMetadata();
    const language = metadata?.get('dc:language');
    const languageCode = Array.isArray(language) ? language[0] : language;
    const encoder = new TextEncoder();
    let text = '', textLimited = false, pagesRead = 0;
    for (let pageNumber = 1; pageNumber <= Math.min(document.numPages, MAX_PAGES) && !textLimited; pageNumber++) {
      const page = await document.getPage(pageNumber);
      pagesRead = pageNumber;
      try {
        const {items} = await page.getTextContent();
        const pageText = items.filter(item => typeof item.str === 'string')
          .map(item => item.str + (item.hasEOL ? '\n' : ' ')).join('').trimStart();
        if (!pageText) continue;
        text += `${text ? '\n\n' : ''}[Page ${pageNumber}]\n${pageText}`;
        const encoded = encoder.encode(text);
        if (encoded.length > MAX_TEXT_BYTES) {
          text = new TextDecoder().decode(encoded.subarray(0, MAX_TEXT_BYTES), {stream: true});
          textLimited = true;
        }
      } finally {
        page.cleanup();
      }
    }
    text = text.trim();
    if (!text)
      throw new Error('This PDF has no extractable text. Scanned or image-only PDFs need OCR, which is not supported yet.');
    const pageLimited = document.numPages > MAX_PAGES && pagesRead === MAX_PAGES && !textLimited;
    const warning = textLimited ? 'Only the first 64 KiB of PDF text was captured.'
      : pageLimited ? 'Only the first 1000 PDF pages were captured.' : '';
    return {
      text, title: typeof info.Title === 'string' ? info.Title.trim() : '',
      language: typeof languageCode === 'string' ? languageCode.trim() : '',
      pageCount: document.numPages,
      truncated: textLimited || pageLimited,
      warning: warning ? `${warning} Results and narration cover the attached text only.` : '',
    };
  } catch (error) {
    if (error.name === 'PasswordException')
      throw new Error('Password-protected PDFs are not supported. Open an unlocked copy or paste its text.');
    throw error;
  } finally {
    try { await loading?.destroy(); }
    finally { worker.destroy(); port.terminate(); }
  }
}
