/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: ["class"],
    content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
  	extend: {
  		fontFamily: {
  			'tesla': ['Gotham', 'Inter', 'system-ui', 'sans-serif'],
  			'tesla-sans': ['Gotham', 'Inter', 'system-ui', 'sans-serif'],
  			'tesla-mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
  		colors: {
  			// Tesla Brand Colors
  			tesla: {
  				red: '#e31e24',
  				black: '#000000',
  				white: '#ffffff',
  				gray: {
  					DEFAULT: '#5c5e62',
  					light: '#f4f4f4',
  					dark: '#3a3a3a'
  				},
  				silver: '#c0c0c0',
  				blue: '#007acc',
  				green: '#00d4aa',
  				orange: '#ff6b35'
  			},
  			// System Colors (Tesla-themed)
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: '#e31e24', // Tesla Red
  				foreground: '#ffffff'
  			},
  			secondary: {
  				DEFAULT: '#5c5e62', // Tesla Gray
  				foreground: '#ffffff'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: '#007acc', // Tesla Blue
  				foreground: '#ffffff'
  			},
  			destructive: {
  				DEFAULT: '#ff6b35', // Tesla Orange
  				foreground: '#ffffff'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': '#e31e24', // Tesla Red
  				'2': '#00d4aa', // Tesla Green
  				'3': '#007acc', // Tesla Blue
  				'4': '#ff6b35', // Tesla Orange
  				'5': '#5c5e62'  // Tesla Gray
  			}
  		},
  		keyframes: {
  			'accordion-down': {
  				from: {
  					height: '0'
  				},
  				to: {
  					height: 'var(--radix-accordion-content-height)'
  				}
  			},
  			'accordion-up': {
  				from: {
  					height: 'var(--radix-accordion-content-height)'
  				},
  				to: {
  					height: '0'
  				}
  			}
  		},
  		animation: {
  			'accordion-down': 'accordion-down 0.2s ease-out',
  			'accordion-up': 'accordion-up 0.2s ease-out'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
};