# ThreatLens-AI

<div align="center">
  <img src="frontend/public/logo.png" alt="ThreatLens-AI Logo" width="200"/>
  <h3>Advanced Terrorism Risk Assessment and Prediction Platform</h3>
</div>

## 📋 Overview

ThreatLens-AI is a sophisticated platform that leverages machine learning and data visualization to forecast potential terrorism hotspots worldwide. By integrating historical terrorism data with socioeconomic indicators and geopolitical factors, the platform provides actionable intelligence for security analysts, researchers, and policymakers.

### Key Features

- 🌐 Interactive 3D globe visualization of terrorism risk
- 📊 Historical data analysis with customizable filters
- 🔮 Predictive modeling for future attack likelihood
- 🧠 Explainable AI to understand prediction factors
- 📱 Responsive design for desktop and mobile devices

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [Python](https://www.python.org/) (v3.11+)
- [MongoDB](https://www.mongodb.com/) (v6+)
- [Neo4j](https://neo4j.com/) (v5+)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/PaulKratsios18/ThreatLens-AI.git
cd ThreatLens-AI
```

2. **Set up the backend**

```bash
cd backend
python -m venv venv_py311
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
pip install -r requirements.txt
```

3. **Set up the frontend**

```bash
cd frontend
npm install
```

4. **Create environment variables**

Copy the example environment file and update with your configuration:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

1. **Start the backend server**

```bash
cd backend
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
python -m uvicorn app.main:app --reload
```

2. **Start the frontend development server**

```bash
cd frontend
npm run dev
```

3. **Access the application**

Open your browser and navigate to `http://localhost:5173`

## 🏗️ Architecture

ThreatLens-AI follows a modern, scalable architecture:

### Frontend

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Library**: Material-UI v5 and Tailwind CSS
- **Data Visualization**: Leaflet.js for 2D maps, Recharts for charts
- **State Management**: React Context API
- **Routing**: React Router v6

### Backend

- **API Framework**: FastAPI with Python 3.11
- **Machine Learning**: TensorFlow, scikit-learn, XGBoost
- **Natural Language Processing**: NLTK
- **Database**: MongoDB for structured data, Neo4j for graph relationships
- **Authentication**: JWT-based auth system

## 📁 Project Structure

```
ThreatLens-AI/
├── backend/              # Python backend code
│   ├── app/              # FastAPI application
│   ├── data_processing/  # Data preparation and ETL
│   ├── models/           # Machine learning models
│   └── scripts/          # Utility scripts
├── frontend/             # React frontend code
│   ├── public/           # Static assets
│   └── src/              # Source code
│       ├── components/   # Reusable UI components
│       ├── contexts/     # React context providers
│       ├── pages/        # Application pages
│       └── utils/        # Utility functions
└── data/                 # Data files and resources
```

## 🧠 Machine Learning Models

The platform uses several ML models to provide accurate predictions:

1. **Geographical Risk Assessment**: Predicts terrorism risk levels for regions based on historical patterns
2. **Attack Type Prediction**: Forecasts the most likely types of attacks for high-risk areas
3. **Temporal Patterns**: Analyzes seasonal and temporal trends in terrorist activities

Models are trained using the Global Terrorism Database (GTD) and supplemented with socioeconomic indicators from the World Bank.

## 📊 Data Sources

- **Global Terrorism Database (GTD)**: Comprehensive dataset of terrorist incidents worldwide
- **World Bank Development Indicators**: Socioeconomic factors by country and region
- **ACLED**: Armed Conflict Location & Event Data Project for recent conflict data
- **GDELT**: Global Database of Events, Language and Tone for media and sentiment analysis

## 🔒 Security

ThreatLens-AI prioritizes security with:

- Data encryption at rest and in transit
- Regular dependency updates to address vulnerabilities
- Input validation and sanitization
- Role-based access control

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

Paul Kratsios - kratsp@rpi.edu

Project Link: [https://github.com/PaulKratsios18/ThreatLens-AI](https://github.com/PaulKratsios18/ThreatLens-AI) 