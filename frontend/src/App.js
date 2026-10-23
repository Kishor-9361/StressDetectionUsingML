import React, { useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import "./theme.css";

function App() {
  useEffect(() => {
    localStorage.removeItem("prism-theme");
    document.body.className = "theme-light";
  }, []);

  return (
    <div className="App theme-light">
      <Dashboard />
    </div>
  );
}

export default App;
