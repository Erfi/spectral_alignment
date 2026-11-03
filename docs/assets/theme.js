// Dark mode toggle. The initial theme is applied by an inline script in <head> to avoid a flash;
// with no saved choice the page follows prefers-color-scheme.
(function () {
  "use strict";

  const root = document.documentElement;
  const button = document.getElementById("theme-toggle");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  function isDark() {
    return root.dataset.theme ? root.dataset.theme === "dark" : systemDark.matches;
  }

  function sync() {
    const dark = isDark();
    button.setAttribute("aria-pressed", String(dark));
    button.title = dark ? "Switch to light mode" : "Switch to dark mode";
  }

  button.addEventListener("click", () => {
    const theme = isDark() ? "light" : "dark";
    root.dataset.theme = theme;
    try { localStorage.setItem("theme", theme); } catch (e) {}
    sync();
  });
  systemDark.addEventListener("change", sync);
  sync();
})();
