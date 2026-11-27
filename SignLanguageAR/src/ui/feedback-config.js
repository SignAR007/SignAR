(() => {
  const localHosts = ["localhost", "127.0.0.1"];
  const productionApiUrl = "";
  window.SignARFeedbackConfig = {
    apiUrl: localHosts.includes(window.location.hostname)
      ? "http://localhost:8787"
      : productionApiUrl,
  };
})();
