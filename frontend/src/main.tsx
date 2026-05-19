import React from "react";
import ReactDOM from "react-dom/client";

// Self-hosted type system. Loaded once at the entry so the rest of the
// app can rely on var(--font-*) without worrying about which weights are
// available. Archivo is variable; Spectral and Fragment Mono ship per-weight.
import "@fontsource-variable/archivo/standard.css";
import "@fontsource-variable/archivo/standard-italic.css";
import "@fontsource/spectral/400.css";
import "@fontsource/spectral/400-italic.css";
import "@fontsource/spectral/500.css";
import "@fontsource/spectral/600.css";
import "@fontsource/fragment-mono/400.css";
import "@fontsource/fragment-mono/400-italic.css";

import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import "./styles/index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
