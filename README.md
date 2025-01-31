# ThreatLens-AI

An advanced terrorism risk assessment and prediction platform that leverages machine learning, knowledge graphs, and diverse data sources to forecast potential terrorism hotspots through interactive 3D visualization.

## 🌟 Features

- **Interactive 3D Globe**: Dark-themed Cesium.js globe with custom atmospheric effects
- **Modern UI**: Material-UI components with responsive design
- **Real-time Analysis**: FastAPI backend for quick data processing
- **Type Safety**: Full TypeScript and Python type support

## 🛠️ Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for fast development
- Cesium.js for 3D visualization
- Material-UI v5 for components
- React Router for navigation

### Backend
- FastAPI with async support
- Python 3.11
- CORS enabled for local development
- MongoDB & Neo4j (planned)

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Node.js 16+
- Git

### Backend Setup

1. Create and activate virtual environment:
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
uvicorn app.main:app --reload
```

Backend runs at http://localhost:8000 with API docs at http://localhost:8000/docs

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm run dev
```

Frontend runs at http://localhost:5173 or http://localhost:5174

## 🗺️ Project Structure

```
ThreatLens-AI/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── api/             # API endpoints
│   │   ├── models/          # Database models
│   │   └── services/        # Business logic
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── globe/       # Cesium globe
│   │   │   └── layout/      # UI layout
│   │   ├── pages/          # Route pages
│   │   └── App.tsx         # Main app
│   ├── package.json        # Node dependencies
│   └── vite.config.ts      # Vite configuration
└── .gitignore             # Git ignore rules
```

## 📊 Data Sources

### Global Terrorism Database (GTD)
- **Source**: University of Maryland's START Consortium (https://www.start.umd.edu/gtd)
- **Access**: Requires academic registration
- **Provides**:
  - Historical terrorism incidents (1970-2022)
  - Attack types and weapons used
  - Target information and casualties
  - Perpetrator groups and their tactics
  - Geographical coordinates of events

### World Bank Development Indicators
- **Source**: World Bank API (https://data.worldbank.org/products/api)
- **Access**: Free, API key required
- **Provides**:
  - GDP and economic indicators
  - Population demographics
  - Education statistics
  - Political stability indices
  - Social development metrics

### OpenStreetMap Data
- **Source**: OpenStreetMap API (https://www.openstreetmap.org)
- **Access**: Free, rate-limited
- **Provides**:
  - Geographic infrastructure data
  - Population density information
  - Building and facility locations
  - Transportation networks
  - Administrative boundaries

### GDELT Project
- **Source**: GDELT API (https://www.gdeltproject.org/)
- **Access**: Free
- **Provides**:
  - Real-time news events
  - Global media coverage
  - Social unrest indicators
  - Political stability metrics
  - Cross-border conflict data

### Data Integration
All data sources are processed through our data pipeline (see `backend/data_processing/`) and stored in:
- MongoDB: Structured event and indicator data
- Neo4j: Knowledge graph relationships between entities

## 🔄 Development Workflow

1. Run both servers:
   - Backend: `uvicorn app.main:app --reload`
   - Frontend: `npm run dev`
2. Backend API is CORS-enabled for frontend ports
3. Changes to code trigger automatic reloads

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 📧 Contact

Paul Kratsios - Paul.Kratsios@gmail.com

Project Link: https://github.com/PaulKratsios18/ThreatLens-AI