# FloatChat - Oceanographic Data Visualization with AI

## Overview

FloatChat is a Streamlit-based web application that combines oceanographic data visualization with AI-powered insights. The application loads oceanographic data from CSV files and uses Google's Gemini AI to provide intelligent analysis and interaction capabilities. It's designed to help researchers and marine scientists visualize and understand oceanographic datasets through an intuitive web interface enhanced with conversational AI.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Visualization**: Plotly Express for interactive data visualization and charts
- **Layout**: Wide layout configuration for better data presentation
- **Data Display**: Pandas DataFrames for structured data manipulation and display

### Backend Architecture
- **Runtime**: Python-based application with modular function design
- **Data Processing**: Pandas for data manipulation and analysis
- **AI Integration**: Google Generative AI (Gemini) for intelligent data insights
- **Configuration Management**: Environment-based API key management for security

### Data Storage
- **Primary Storage**: CSV file-based data storage (`data.csv`)
- **Caching Strategy**: Streamlit's `@st.cache_data` decorator for performance optimization
- **Data Format**: Structured oceanographic datasets in tabular format

### Authentication and Security
- **API Security**: Environment variable-based API key management for Gemini
- **Error Handling**: Comprehensive error handling for missing files and API failures
- **Graceful Degradation**: Application stops with clear error messages when dependencies fail

## External Dependencies

### AI Services
- **Google Generative AI (Gemini)**: Provides conversational AI capabilities for data analysis and insights
- **API Key Requirement**: Requires `GEMINI_API_KEY` environment variable

### Python Libraries
- **Streamlit**: Web application framework for data science applications
- **Pandas**: Data manipulation and analysis library
- **Plotly Express**: Interactive visualization library for creating charts and graphs
- **OS**: System environment variable access

### Data Dependencies
- **CSV Data Source**: Requires `data.csv` file containing oceanographic measurements
- **Data Format**: Expects structured tabular data compatible with Pandas DataFrame operations

### Development Environment
- **Python Runtime**: Compatible with modern Python versions
- **Package Management**: Standard pip-installable packages
- **Environment Configuration**: Relies on environment variables for external service authentication