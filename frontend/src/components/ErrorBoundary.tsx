import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[LabVerse] Uncaught error:", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error !== null) {
      return (
        <div className="fatal">
          <div className="fatal__sheet">
            <p className="kicker">Application error</p>
            <h1 className="title">The instrument has stopped responding.</h1>
            <p className="fatal__detail">{this.state.error.message}</p>
            <div className="flex-row mt-4">
              <button
                className="btn-accent"
                onClick={() => {
                  this.setState({ error: null });
                  window.location.reload();
                }}
                type="button"
              >
                Reload
              </button>
              <button
                className="btn-ghost"
                onClick={() => this.setState({ error: null })}
                type="button"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
