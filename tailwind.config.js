import forms from '@tailwindcss/forms';
import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        /* Desaturated, serious palette */
        primary: "#546a7b",
        "primary-hover": "#455a6a",
        "primary-muted": "#6b7c8a",
        "background-light": "#f5f6f7",
        "background-dark": "#151b23",
        "card-light": "#ffffff",
        "card-dark": "#1e262f",
        "border-light": "#c5cdd4",
        "border-dark": "#20272f",
        "accent-muted": "#5a6b7a",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.5rem",
      },
    },
  },
  plugins: [forms, typography],
}
