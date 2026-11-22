const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const themeSource = fs.readFileSync(
  path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'ui', 'theme.js'),
  'utf8'
);

function createThemeEnvironment(initialPreference, systemPrefersDark = false) {
  const root = { dataset: {} };
  const themeColor = { content: '' };
  const listeners = {};
  const buttonListeners = {};
  const button = {
    textContent: '',
    title: '',
    attributes: {},
    addEventListener: (name, callback) => { buttonListeners[name] = callback; },
    setAttribute: (name, value) => { button.attributes[name] = value; }
  };
  let savedPreference = initialPreference;
  const document = {
    documentElement: root,
    querySelector: () => themeColor,
    addEventListener: (name, callback) => { listeners[name] = callback; },
    getElementById: id => id === 'themeToggle' ? button : null
  };
  const window = {
    localStorage: {
      getItem: () => savedPreference,
      setItem: (key, value) => { savedPreference = value; }
    },
    matchMedia: () => ({ matches: systemPrefersDark }),
    addEventListener: (name, callback) => { listeners[name] = callback; }
  };

  vm.runInNewContext(themeSource, {
    window,
    document,
    console: { warn() {}, error() {} }
  });
  listeners.DOMContentLoaded();
  return { root, themeColor, button, buttonListeners, listeners, getSavedPreference: () => savedPreference };
}

const defaultTheme = createThemeEnvironment(null, true);
assert.equal(defaultTheme.root.dataset.theme, 'light', 'a dark system preference should not override the light default');
assert.equal(defaultTheme.themeColor.content, '#f1f6fb');
assert.equal(defaultTheme.button.textContent, 'Dark mode');

defaultTheme.buttonListeners.click();
assert.equal(defaultTheme.root.dataset.theme, 'dark', 'users should still be able to switch to dark mode');
assert.equal(defaultTheme.getSavedPreference(), 'dark', 'explicit theme choices should remain saved');

defaultTheme.listeners.storage({ key: 'signar-theme', newValue: null });
assert.equal(defaultTheme.root.dataset.theme, 'light', 'clearing a saved choice should restore the light default');

const savedDarkTheme = createThemeEnvironment('dark');
assert.equal(savedDarkTheme.root.dataset.theme, 'dark', 'an explicit saved dark preference should be honored');
assert.equal(savedDarkTheme.button.textContent, 'Light mode');
