// Dark (default), light or system theme, remembered per browser; the top-bar button cycles it.
import {Theme} from './liteapp/ide/theme.mjs';

export const theme = new Theme({storageKey: 'kidi-theme', button: document.getElementById('theme')});
