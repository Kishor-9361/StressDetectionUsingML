import React from "react";
import { Link, useLocation } from "react-router-dom";
import "../theme.css";

export default function Navbar() {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="navbar">
      <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Link className="navbar-brand" to="/">
          <span style={{ fontSize: '1.5em' }}>🧠</span> Stress Analytics
        </Link>

        {/* Desktop Menu */}
        <div className="navbar-nav-desktop" style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <Link className={`nav-link ${isActive('/') ? 'active' : ''}`} to="/">Home</Link>
          <Link className={`nav-link ${isActive('/about') ? 'active' : ''}`} to="/about">About</Link>
          <Link className={`nav-link ${isActive('/features') ? 'active' : ''}`} to="/features">Features</Link>
          <Link className={`nav-link ${isActive('/impact') ? 'active' : ''}`} to="/impact">Impact</Link>
          <Link
            className="btn btn-neon"
            to="/dashboard"
            style={{ fontWeight: 600, padding: '0.6rem 1.5rem', color: '#111' }}
          >
            Launch Dashboard
          </Link>
        </div>
      </div>
    </nav>
  );
}