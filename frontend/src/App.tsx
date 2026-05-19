import PaperView from "./components/paper/PaperView";

export default function App(): JSX.Element {
  return (
    <div className="app app--paper-shell">
      <main className="main main--paper">
        <PaperView />
      </main>
    </div>
  );
}
