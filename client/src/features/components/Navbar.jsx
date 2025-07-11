import { useState } from "react";
import { Link } from "react-router-dom";
import { Menu, X } from "lucide-react";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-[#121212] text-[#E0E0E0] p-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        {/* Logo */}
        <Link to="/" className="text-2xl font-bold text-[#BB86FC]">
          NeuroScanAI
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex gap-6">
          <NavLink to="/" label="Home" />
          <NavLink to="/blogs" label="Blogs" />
          <NavLink to="/hospitals" label="Hospitals" />
          <NavLink to="/detect" label="Detection" />
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden flex flex-col mt-3 space-y-2 bg-[#1E1E1E] p-4 rounded-lg">
          <NavLink to="/" label="Home" onClick={() => setMenuOpen(false)} />
          <NavLink to="/blogs" label="Blogs" onClick={() => setMenuOpen(false)} />
          <NavLink to="/hospitals" label="Hospitals" onClick={() => setMenuOpen(false)} />
          <NavLink to="/detect" label="Detection" onClick={() => setMenuOpen(false)} />
        </div>
      )}
    </nav>
  );
}

// Reusable NavLink Component
function NavLink({ to, label, onClick }) {
  return (
    <Link
      to={to}
      onClick={onClick}
      className="block md:inline-block px-3 py-2 rounded-lg text-[#E0E0E0] hover:bg-[#BB86FC] hover:text-[#121212] transition"
    >
      {label}
    </Link>
  );
}
