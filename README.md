# ThreatLens-AI

An advanced terrorism risk assessment and prediction platform that leverages machine learning, knowledge graphs, and diverse data sources to forecast potential terrorism hotspots through interactive 3D visualization.

## 🌟 Features

- **Advanced Risk Assessment**: Utilizes machine learning models to predict potential terrorism hotspots
- **Interactive 3D Visualization**: Explore data through an immersive Cesium.js-powered globe interface
- **Knowledge Graph Integration**: Connects complex relationships between events, actors, and socio-economic factors
- **Multi-Source Data Analysis**: Combines structured, unstructured, and geospatial data for comprehensive insights
- **Real-time Updates**: Incorporates live news and OSINT data for current threat assessment

## 🛠️ Tech Stack

- **Frontend**: React.js, Cesium.js
- **Backend**: Python, FastAPI
- **Databases**: MongoDB, Neo4j
- **ML/AI**: scikit-learn, XGBoost, NLTK
- **Data Processing**: Pandas, NumPy
- **APIs**: GTD, World Bank, NewsAPI

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB
- Neo4j

### Installation

1. Clone the repository
```bash
git clone https://github.com/PaulKratsios18/ThreatLens-AI.git
cd ThreatLens-AI
```

2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

3. Install frontend dependencies
```bash
cd frontend
npm install
```

4. Set up environment variables
```bash
cp .env.example .env
```

5. Run the application
```bash
# Backend
python main.py

# Frontend
npm run dev
```

## 📊 Data Sources

- Global Terrorism Database (GTD)
- World Bank Indicators
- OSINT and News APIs
- OpenStreetMap
- Social Media Data

## 🗺️ Project Structure

```
ThreatLens-AI/
├── backend/
│   ├── data_processing/
│   ├── ml_models/
│   ├── knowledge_graph/
│   └── api/
├── frontend/
│   ├── components/
│   ├── pages/
│   └── visualization/
└── data/
    ├── raw/
    └── processed/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Paul Kratsios - Paul.Kratsios@gmail.com

Project Link: https://github.com/PaulKratsios18/ThreatLens-AI