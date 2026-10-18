const key = value => value.normalize('NFKC').trim().toLocaleLowerCase();

export function parseLanguages(text) {
  const aliases = new Map();
  const languages = [];
  for (const line of text.split(/\r?\n/u)) {
    if (!line || line.startsWith('#')) continue;
    const [code, name, ...names] = line.split('\t');
    if (!/^[a-z]{2,3}$/u.test(code) || !name) throw new Error('Invalid language table');
    const language = {code, name, aliases: [code, name, ...names].map(key)};
    languages.push(language);
    for (const alias of [code, name, ...names]) aliases.set(key(alias), code);
  }
  return {aliases, languages};
}

export function resolveLanguages(value, locale, aliases) {
  const requested = String(value || 'auto').split(',').map(key).filter(Boolean);
  const local = key(String(locale || 'en').split('-')[0]);
  const resolved = requested.map(language => aliases.get(language === 'auto' ? local : language))
    .filter(Boolean);
  return [...new Set(resolved.length ? resolved : [aliases.get('english') || 'en'])].slice(0, 8);
}

export const resolveLanguage = (value, aliases) => aliases.get(key(String(value || ''))) || '';

export const activeLanguages = (configured, active) =>
  active && configured.includes(active) ? [active] : configured;
