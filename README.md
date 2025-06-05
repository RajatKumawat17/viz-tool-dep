# Automated Data Visualization Tool

A web-based tool for automated data analysis and visualization. Upload CSV files and get instant insights with beautiful charts and downloadable reports.

## Features

- **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- **Automatic Analysis**: Generates insights about your data structure
- **Multiple Chart Types**: Distribution plots, categorical charts, correlation matrices, scatter plots
- **Interactive Visualizations**: Responsive charts with hover effects
- **Downloadable Reports**: Generate HTML reports with all insights and chart descriptions
- **Mobile Responsive**: Works on all devices

## Supported File Formats

- CSV files (.csv)
- Excel files (.xlsx, .xls) - *Basic support*

## Quick Start

### Local Development

1. **Clone/Download** the project files
2. **Open** `index.html` in your browser
3. **Upload** a CSV file to start analyzing

### Deploy to Netlify

1. **Create a new site** on Netlify
2. **Upload** all project files or connect your Git repository
3. **Deploy** - the site will be live instantly

## File Structure

```
project/
├── index.html          # Main application
├── netlify.toml        # Netlify configuration
├── package.json        # Project metadata
└── README.md          # This file
```

## Usage

1. Open the application in your browser
2. Drag and drop your CSV file or click to browse
3. Wait for the analysis to complete
4. Review the generated insights and visualizations
5. Download the complete HTML report

## Technical Details

- **Frontend**: Pure HTML, CSS, JavaScript
- **Charts**: Chart.js for static charts, Plotly for interactive elements
- **CSV Parsing**: Papa Parse library
- **Hosting**: Netlify (static hosting)
- **No Backend**: Runs entirely in the browser

## Limitations

- File size limit depends on browser memory (typically 50-100MB)
- Excel support is basic (CSV recommended)
- No data persistence (analysis is temporary)
- Runs client-side only

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Contributing

Feel free to submit issues and pull requests to improve the tool.

## License

MIT License - see package.json for details