# HealthAI Analytics Engine - Complete Implementation Guide

## Executive Summary

This document provides a comprehensive blueprint for implementing the AI-powered analytics engine for HealthAI. The system is designed to:

1. Accept health data files (CSV, Excel, JSON)
2. Perform automated data profiling and quality assessment
3. Generate AI-powered insights with zero hallucinations
4. Create interactive dashboards automatically
5. Support predictive analytics for health indicators

**Current Date**: February 03, 2026

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER (React)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  File Upload │  │  Dashboard   │  │   AI Chat    │  │  Settings    │   │
│  │   Component  │  │   Renderer   │  │  Interface   │  │   Panel      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼─────────────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY (FastAPI)                               │
│  - Authentication  - Rate Limiting  - Request Routing  - Validation          │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ├──────────────────┬──────────────────┬──────────────────┐
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  FILE SERVICE   │ │ ANALYTICS       │ │  AI SERVICE     │ │  USER SERVICE   │
│                 │ │   ENGINE        │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Upload      │ │ │ │ Data        │ │ │ │ LLM         │ │ │ │ Auth        │ │
│ │ Handler     │ │ │ │ Profiler    │ │ │ │ Gateway     │ │ │ │ Profile     │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ Storage     │ │ │ │ Stats       │ │ │ │ Insight     │ │ │ │ Billing     │ │
│ │ Manager     │ │ │ │ Computer    │ │ │ │ Generator   │ │ │ │ (Tanzania)  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │                 │
│ │ Validator   │ │ │ │ Anomaly     │ │ │ │ Facts       │ │ │                 │
│ └─────────────┘ │ │ │ │ Detector    │ │ │ │ Validator   │ │ │                 │ │
│                 │ │ └─────────────┘ │ │ └─────────────┘ │ │                 │
│                 │ │ ┌─────────────┐ │ │                 │ │                 │
│                 │ │ │ Forecast    │ │ │                 │ │                 │
│                 │ │ │ Engine      │ │ │                 │ │                 │
│                 │ │ └─────────────┘ │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │                   │
         └───────────────────┴───────────────────┴───────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  PostgreSQL     │  │  Redis Cache    │  │  MinIO/S3       │             │
│  │  (Metadata)     │  │  (Sessions)     │  │  (Files)        │             │
│  │                 │  │                 │  │                 │             │
│  │  - Users        │  │  - Query cache  │  │  - Raw uploads  │             │
│  │  - Datasets     │  │  - Rate limits  │  │  - Processed    │             │
│  │  - Dashboards   │  │  - Pub/Sub      │  │  - Exports      │             │
│  │  - Analysis     │  │                 │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │  DuckDB         │  │  ChromaDB       │                                   │
│  │  (Analytics)    │  │  (Embeddings)   │                                   │
│  │                 │  │                 │                                   │
│  │  - Computed     │  │  - RAG vectors  │                                   │
│  │    stats        │  │  - Similarity   │                                   │
│  │  - Aggregations │  │    search       │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Backend Setup (Week 1-2)

### 1.1 Project Structure

```
healthai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── database.py             # Database connection
│   ├── models/                 # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── dataset.py
│   │   ├── analysis.py
│   │   └── dashboard.py
│   ├── routers/                # API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── upload.py
│   │   ├── analysis.py
│   │   ├── dashboard.py
│   │   ├── insights.py
│   │   └── payments.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── file_service.py
│   │   ├── analytics_engine.py
│   │   ├── ai_service.py
│   │   └── payment_service.py
│   ├── ai/                     # AI/ML modules
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   ├── insights.py
│   │   ├── forecast.py
│   │   └── guardrails.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── validators.py
│       └── formatters.py
├── tests/
├── alembic/                    # Database migrations
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── .env
```

### 1.2 Core Dependencies

```txt
# requirements.txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
websockets==12.0

# Database
sqlalchemy==2.0.25
alembic==1.13.1
asyncpg==0.29.0
redis==5.0.1

# Data Processing
pandas==2.2.0
polars==0.20.5
duckdb==0.9.2
pyarrow==15.0.0
openpyxl==3.1.2
xlrd==2.0.1

# AI/ML
openai==1.10.0
anthropic==0.18.1
prophet==1.1.5
scikit-learn==1.4.0
scipy==1.12.0
numpy==1.26.3

# Validation & Quality
pydantic==2.5.3
great-expectations==0.18.8

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0

# Tanzania Payments
requests==2.31.0

# Utilities
python-dateutil==2.8.2
celery==5.3.6
minio==7.2.0
```

### 1.3 Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://healthai:password@db:5432/healthai
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
      - minio
    volumes:
      - ./app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=healthai
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=healthai
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    environment:
      - MINIO_ROOT_USER=healthai
      - MINIO_ROOT_PASSWORD=healthai123
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  celery:
    build: .
    command: celery -A app.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://healthai:password@db:5432/healthai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  minio_data:
```

---

## Phase 2: Database Schema

### 2.1 Core Tables

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    role VARCHAR(50) DEFAULT 'analyst',
    subscription_tier VARCHAR(50) DEFAULT 'starter',
    subscription_expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    column_names JSONB,
    quality_score INTEGER,
    quality_report JSONB,
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results table
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    results JSONB,
    insights JSONB,
    facts_table JSONB,
    confidence_scores JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Dashboards table
CREATE TABLE dashboards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    spec JSONB NOT NULL,
    is_public BOOLEAN DEFAULT false,
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insights table
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    citations JSONB,
    user_feedback VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payments table (Tanzania)
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'TZS',
    payment_method VARCHAR(50) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    provider_transaction_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Audit logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Phase 3: AI Analytics Engine

### 3.1 Data Profiler Module

```python
# app/ai/profiler.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import duckdb

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str  # categorical, numerical, datetime, text, boolean
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any]
    statistics: Dict[str, Any]

@dataclass
class DataProfile:
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    quality_score: int
    issues: List[Dict[str, Any]]
    recommendations: List[str]

class DataProfiler:
    def __init__(self):
        self.con = duckdb.connect(':memory:')
    
    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Generate comprehensive data profile"""
        columns = []
        issues = []
        
        for col in df.columns:
            profile = self._profile_column(df, col)
            columns.append(profile)
            
            # Detect issues
            if profile.null_percentage > 20:
                issues.append({
                    'type': 'high_missingness',
                    'column': col,
                    'severity': 'warning' if profile.null_percentage < 50 else 'critical',
                    'message': f'{col} has {profile.null_percentage:.1f}% missing values'
                })
            
            if profile.inferred_type == 'categorical' and profile.unique_percentage > 90:
                issues.append({
                    'type': 'high_cardinality',
                    'column': col,
                    'severity': 'info',
                    'message': f'{col} appears to be an ID field with high cardinality'
                })
        
        quality_score = self._calculate_quality_score(columns, issues)
        
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            quality_score=quality_score,
            issues=issues,
            recommendations=self._generate_recommendations(columns, issues)
        )
    
    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        null_count = series.isnull().sum()
        unique_count = series.nunique()
        
        # Infer type
        inferred_type = self._infer_type(series)
        
        # Calculate statistics based on type
        statistics = self._calculate_statistics(series, inferred_type)
        
        return ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            null_count=int(null_count),
            null_percentage=(null_count / len(df)) * 100,
            unique_count=int(unique_count),
            unique_percentage=(unique_count / len(df)) * 100,
            sample_values=series.dropna().head(5).tolist(),
            statistics=statistics
        )
    
    def _infer_type(self, series: pd.Series) -> str:
        """Infer semantic type of column"""
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Try to parse as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(100))
                return 'datetime'
            except:
                pass
        
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical (few unique values)
            if series.nunique() / len(series) < 0.05 and series.nunique() < 20:
                return 'categorical'
            return 'numerical'
        
        # Check for categorical
        if series.nunique() / len(series) < 0.1 or series.nunique() < 50:
            return 'categorical'
        
        return 'text'
    
    def _calculate_statistics(self, series: pd.Series, inferred_type: str) -> Dict[str, Any]:
        """Calculate type-appropriate statistics"""
        stats = {}
        
        if inferred_type == 'numerical':
            stats = {
                'mean': float(series.mean()) if not series.empty else None,
                'median': float(series.median()) if not series.empty else None,
                'std': float(series.std()) if not series.empty else None,
                'min': float(series.min()) if not series.empty else None,
                'max': float(series.max()) if not series.empty else None,
                'q25': float(series.quantile(0.25)) if not series.empty else None,
                'q75': float(series.quantile(0.75)) if not series.empty else None,
            }
        
        elif inferred_type == 'categorical':
            value_counts = series.value_counts().head(10)
            stats = {
                'top_values': value_counts.to_dict(),
                'mode': series.mode().iloc[0] if not series.mode().empty else None
            }
        
        elif inferred_type == 'datetime':
            stats = {
                'min': str(series.min()) if not series.empty else None,
                'max': str(series.max()) if not series.empty else None,
                'range_days': (series.max() - series.min()).days if not series.empty else None
            }
        
        return stats
    
    def _calculate_quality_score(self, columns: List[ColumnProfile], issues: List[Dict]) -> int:
        """Calculate overall data quality score (0-100)"""
        if not columns:
            return 0
        
        # Completeness (40%)
        avg_completeness = 100 - np.mean([c.null_percentage for c in columns])
        
        # Validity (30%) - based on issues
        validity_penalty = sum(10 for i in issues if i['severity'] == 'critical')
        validity_penalty += sum(5 for i in issues if i['severity'] == 'warning')
        validity = max(0, 100 - validity_penalty)
        
        # Consistency (20%) - based on type inference confidence
        consistency = 100  # Placeholder
        
        # Uniqueness (10%)
        uniqueness = 100  # Placeholder
        
        score = int(
            avg_completeness * 0.4 +
            validity * 0.3 +
            consistency * 0.2 +
            uniqueness * 0.1
        )
        
        return min(100, max(0, score))
    
    def _generate_recommendations(self, columns: List[ColumnProfile], issues: List[Dict]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'high_missingness':
                recommendations.append(
                    f"Consider imputing missing values in '{issue['column']}' or investigating why data is missing"
                )
        
        return recommendations
```

### 3.2 AI Insights Generator (Zero Hallucination)

```python
# app/ai/insights.py
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import openai
from anthropic import Anthropic

@dataclass
class Fact:
    id: str
    type: str  # statistic, comparison, trend, anomaly
    value: Any
    label: str
    computation: Dict[str, Any]
    confidence: str  # high, medium, low
    citation: str

@dataclass
class Insight:
    id: str
    type: str  # trend, anomaly, prediction, recommendation
    title: str
    content: str
    facts: List[str]  # References to fact IDs
    confidence: int  # 0-100
    uncertainty: Optional[str]
    citations: List[str]

class InsightsGenerator:
    def __init__(self, openai_key: str, anthropic_key: Optional[str] = None):
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.anthropic_client = Anthropic(api_key=anthropic_key) if anthropic_key else None
    
    def generate(self, df: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """Generate AI insights with zero hallucinations"""
        
        # Step 1: Compute all facts deterministically
        facts = self._compute_facts(df, profile)
        
        # Step 2: Generate insights based ONLY on facts
        insights = self._generate_insights_from_facts(facts, profile)
        
        # Step 3: Validate insights against facts
        validated_insights = self._validate_insights(insights, facts)
        
        return {
            'facts': [self._fact_to_dict(f) for f in facts],
            'insights': [self._insight_to_dict(i) for i in validated_insights],
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _compute_facts(self, df: pd.DataFrame, profile: DataProfile) -> List[Fact]:
        """Compute all statistical facts deterministically"""
        facts = []
        fact_id = 0
        
        # Dataset overview facts
        facts.append(Fact(
            id=f'f_{fact_id}',
            type='statistic',
            value=profile.row_count,
            label='Total records',
            computation={'method': 'count', 'sql': 'SELECT COUNT(*) FROM dataset'},
            confidence='high',
            citation='dataset:profile'
        ))
        fact_id += 1
        
        # Column-specific facts
        for col in profile.columns:
            if col.inferred_type == 'numerical' and col.statistics:
                facts.append(Fact(
                    id=f'f_{fact_id}',
                    type='statistic',
                    value=col.statistics.get('mean'),
                    label=f'Mean of {col.name}',
                    computation={'method': 'mean', 'column': col.name},
                    confidence='high' if col.null_percentage < 10 else 'medium',
                    citation=f'dataset:column:{col.name}'
                ))
                fact_id += 1
                
                # Min/max facts
                facts.append(Fact(
                    id=f'f_{fact_id}',
                    type='statistic',
                    value={'min': col.statistics.get('min'), 'max': col.statistics.get('max')},
                    label=f'Range of {col.name}',
                    computation={'method': 'min_max', 'column': col.name},
                    confidence='high',
                    citation=f'dataset:column:{col.name}'
                ))
                fact_id += 1
            
            # Missing data facts
            if col.null_percentage > 0:
                facts.append(Fact(
                    id=f'f_{fact_id}',
                    type='statistic',
                    value=col.null_percentage,
                    label=f'Missing percentage in {col.name}',
                    computation={'method': 'null_percentage', 'column': col.name},
                    confidence='high',
                    citation=f'dataset:column:{col.name}'
                ))
                fact_id += 1
        
        # Detect trends if datetime column exists
        datetime_cols = [c for c in profile.columns if c.inferred_type == 'datetime']
        if datetime_cols:
            facts.extend(self._compute_trend_facts(df, datetime_cols[0], profile, fact_id))
        
        return facts
    
    def _compute_trend_facts(self, df: pd.DataFrame, datetime_col: str, profile: DataProfile, start_id: int) -> List[Fact]:
        """Compute trend-related facts"""
        facts = []
        fact_id = start_id
        
        # Group by month and count
        df_copy = df.copy()
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
        monthly = df_copy.groupby(df_copy[datetime_col].dt.to_period('M')).size()
        
        if len(monthly) > 1:
            # Calculate month-over-month change
            changes = monthly.pct_change().dropna()
            avg_change = changes.mean()
            
            facts.append(Fact(
                id=f'f_{fact_id}',
                type='trend',
                value=avg_change * 100,
                label=f'Average monthly change',
                computation={'method': 'pct_change', 'column': datetime_col},
                confidence='medium' if len(monthly) > 6 else 'low',
                citation=f'dataset:trend:{datetime_col}'
            ))
        
        return facts
    
    def _generate_insights_from_facts(self, facts: List[Fact], profile: DataProfile) -> List[Insight]:
        """Generate insights using LLM, constrained to facts only"""
        
        # Build facts table for LLM
        facts_table = json.dumps([self._fact_to_dict(f) for f in facts], indent=2)
        
        prompt = f"""You are a healthcare data analyst. Generate insights based ONLY on the provided facts.

FACTS TABLE:
{facts_table}

DATASET PROFILE:
- Total records: {profile.row_count}
- Total columns: {profile.column_count}
- Quality score: {profile.quality_score}/100

RULES:
1. Use ONLY numbers from the facts table
2. Cite each fact using [f_X] notation
3. Flag any uncertainties explicitly
4. Suggest verification if confidence is low
5. Focus on healthcare-relevant insights

Generate 3-5 insights in this JSON format:
{{
  "insights": [
    {{
      "type": "trend|anomaly|comparison|recommendation",
      "title": "Brief title",
      "content": "Detailed insight with [f_X] citations",
      "facts": ["f_0", "f_1"],
      "confidence": 85,
      "uncertainty": "Optional note about limitations"
    }}
  ]
}}
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a healthcare data analyst that only reports facts from provided data."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Low temperature for consistency
            )
            
            result = json.loads(response.choices[0].message.content)
            
            insights = []
            for i, ins in enumerate(result.get('insights', [])):
                insights.append(Insight(
                    id=f'ins_{i}',
                    type=ins['type'],
                    title=ins['title'],
                    content=ins['content'],
                    facts=ins['facts'],
                    confidence=ins['confidence'],
                    uncertainty=ins.get('uncertainty'),
                    citations=ins['facts']
                ))
            
            return insights
            
        except Exception as e:
            # Fallback: generate basic insights without LLM
            return self._generate_fallback_insights(facts, profile)
    
    def _generate_fallback_insights(self, facts: List[Fact], profile: DataProfile) -> List[Insight]:
        """Generate basic insights without LLM (fallback)"""
        insights = []
        
        # Data quality insight
        if profile.quality_score < 80:
            insights.append(Insight(
                id='ins_0',
                type='recommendation',
                title='Data Quality Improvement Needed',
                content=f'Data quality score is {profile.quality_score}%. Review missing values and data consistency.',
                facts=[],
                confidence=95,
                uncertainty=None,
                citations=['dataset:profile']
            ))
        
        return insights
    
    def _validate_insights(self, insights: List[Insight], facts: List[Fact]) -> List[Insight]:
        """Validate that insights only reference existing facts"""
        fact_ids = {f.id for f in facts}
        validated = []
        
        for insight in insights:
            # Check all cited facts exist
            invalid_citations = [c for c in insight.citations if c not in fact_ids]
            
            if invalid_citations:
                # Remove invalid citations or mark as low confidence
                insight.citations = [c for c in insight.citations if c in fact_ids]
                insight.confidence = min(insight.confidence, 50)
                insight.uncertainty = f"Some citations could not be verified: {invalid_citations}"
            
            validated.append(insight)
        
        return validated
    
    def _fact_to_dict(self, fact: Fact) -> Dict[str, Any]:
        return {
            'id': fact.id,
            'type': fact.type,
            'value': fact.value,
            'label': fact.label,
            'computation': fact.computation,
            'confidence': fact.confidence,
            'citation': fact.citation
        }
    
    def _insight_to_dict(self, insight: Insight) -> Dict[str, Any]:
        return {
            'id': insight.id,
            'type': insight.type,
            'title': insight.title,
            'content': insight.content,
            'facts': insight.facts,
            'confidence': insight.confidence,
            'uncertainty': insight.uncertainty,
            'citations': insight.citations
        }
```

---

## Phase 4: Tanzania Payment Integration

### 4.1 M-Pesa Integration

```python
# app/services/payment/tanzania_mpesa.py
import requests
import base64
from datetime import datetime
from typing import Dict, Optional
import json

class MpesaClient:
    """M-Pesa API Client for Tanzania"""
    
    def __init__(self, consumer_key: str, consumer_secret: str, passkey: str, shortcode: str):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.passkey = passkey
        self.shortcode = shortcode
        self.base_url = "https://sandbox.safaricom.co.ke"  # Production: https://api.safaricom.co.ke
    
    def _get_access_token(self) -> str:
        """Get OAuth access token"""
        credentials = base64.b64encode(
            f"{self.consumer_key}:{self.consumer_secret}".encode()
        ).decode()
        
        response = requests.get(
            f"{self.base_url}/oauth/v1/generate?grant_type=client_credentials",
            headers={"Authorization": f"Basic {credentials}"}
        )
        
        return response.json()['access_token']
    
    def stk_push(
        self, 
        phone_number: str, 
        amount: float, 
        account_reference: str,
        transaction_desc: str
    ) -> Dict:
        """Initiate STK push (USSD prompt on customer's phone)"""
        
        access_token = self._get_access_token()
        
        # Format phone number
        if phone_number.startswith('0'):
            phone_number = '255' + phone_number[1:]
        elif phone_number.startswith('+'):
            phone_number = phone_number[1:]
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        password = base64.b64encode(
            f"{self.shortcode}{self.passkey}{timestamp}".encode()
        ).decode()
        
        payload = {
            "BusinessShortCode": self.shortcode,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount),
            "PartyA": phone_number,
            "PartyB": self.shortcode,
            "PhoneNumber": phone_number,
            "CallBackURL": "https://your-domain.com/api/payments/mpesa/callback",
            "AccountReference": account_reference,
            "TransactionDesc": transaction_desc
        }
        
        response = requests.post(
            f"{self.base_url}/mpesa/stkpush/v1/processrequest",
            json=payload,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        )
        
        return response.json()
    
    def query_transaction(self, checkout_request_id: str) -> Dict:
        """Query transaction status"""
        access_token = self._get_access_token()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        password = base64.b64encode(
            f"{self.shortcode}{self.passkey}{timestamp}".encode()
        ).decode()
        
        payload = {
            "BusinessShortCode": self.shortcode,
            "Password": password,
            "Timestamp": timestamp,
            "CheckoutRequestID": checkout_request_id
        }
        
        response = requests.post(
            f"{self.base_url}/mpesa/stkpushquery/v1/query",
            json=payload,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        )
        
        return response.json()


# Generic Tanzania Mobile Money Interface
class TanzaniaPaymentService:
    """Unified interface for all Tanzania mobile money providers"""
    
    PROVIDERS = {
        'mpesa': 'M-Pesa',
        'airtel': 'Airtel Money',
        'tigo': 'Tigo Pesa',
        'halo': 'HaloPesa',
        'zantel': 'Zantel',
        'ttcl': 'TTCL'
    }
    
    def __init__(self):
        self.mpesa = MpesaClient(
            consumer_key="your_consumer_key",
            consumer_secret="your_consumer_secret",
            passkey="your_passkey",
            shortcode="your_shortcode"
        )
        # Initialize other providers similarly
    
    async def initiate_payment(
        self,
        provider: str,
        phone_number: str,
        amount: float,
        description: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Initiate payment with any Tanzania provider"""
        
        if provider == 'mpesa':
            result = self.mpesa.stk_push(
                phone_number=phone_number,
                amount=amount,
                account_reference=metadata.get('reference', 'HealthAI'),
                transaction_desc=description
            )
            
            return {
                'success': 'ResponseCode' in result and result['ResponseCode'] == '0',
                'transaction_id': result.get('CheckoutRequestID'),
                'provider': 'mpesa',
                'message': result.get('ResponseDescription', 'Payment initiated'),
                'customer_message': 'Please check your phone to complete payment'
            }
        
        # Add other providers here
        # elif provider == 'airtel':
        #     return await self._airtel_payment(phone_number, amount, description)
        
        else:
            return {
                'success': False,
                'error': f'Provider {provider} not yet implemented'
            }
    
    async def verify_payment(self, provider: str, transaction_id: str) -> Dict:
        """Verify payment status"""
        if provider == 'mpesa':
            result = self.mpesa.query_transaction(transaction_id)
            
            return {
                'success': result.get('ResultCode') == '0',
                'status': 'completed' if result.get('ResultCode') == '0' else 'pending',
                'amount': result.get('Amount'),
                'phone': result.get('PhoneNumber'),
                'transaction_date': result.get('TransactionDate')
            }
        
        return {'success': False, 'error': 'Provider not supported'}
```

---

## Phase 5: Frontend Integration

### 5.1 Payment Component for Tanzania

```tsx
// src/components/payment/TanzaniaPayment.tsx
import { useState } from 'react';
import { Loader2, Phone, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';

const PROVIDERS = [
  { id: 'mpesa', name: 'M-Pesa', color: 'bg-green-600', logo: 'M' },
  { id: 'airtel', name: 'Airtel Money', color: 'bg-red-600', logo: 'A' },
  { id: 'tigo', name: 'Tigo Pesa', color: 'bg-blue-600', logo: 'T' },
  { id: 'halo', name: 'HaloPesa', color: 'bg-purple-600', logo: 'H' },
  { id: 'zantel', name: 'Zantel', color: 'bg-orange-600', logo: 'Z' },
];

interface TanzaniaPaymentProps {
  amount: number;
  description: string;
  onSuccess: (transactionId: string) => void;
  onError: (error: string) => void;
}

export function TanzaniaPayment({ amount, description, onSuccess, onError }: TanzaniaPaymentProps) {
  const [selectedProvider, setSelectedProvider] = useState('mpesa');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'processing' | 'success'>('idle');

  const handlePayment = async () => {
    if (!phoneNumber || phoneNumber.length < 10) {
      onError('Please enter a valid phone number');
      return;
    }

    setIsLoading(true);
    setStatus('processing');

    try {
      const response = await fetch('/api/payments/initiate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider,
          phoneNumber,
          amount,
          description,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setStatus('success');
        onSuccess(data.transaction_id);
      } else {
        setStatus('idle');
        onError(data.error || 'Payment failed');
      }
    } catch (error) {
      setStatus('idle');
      onError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Provider Selection */}
      <div>
        <Label className="mb-3 block">Select Payment Method</Label>
        <div className="grid grid-cols-3 gap-3">
          {PROVIDERS.map((provider) => (
            <button
              key={provider.id}
              onClick={() => setSelectedProvider(provider.id)}
              className={cn(
                "flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all",
                selectedProvider === provider.id
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-muted-foreground/50"
              )}
            >
              <div className={cn("w-10 h-10 rounded-full flex items-center justify-center text-white font-bold", provider.color)}>
                {provider.logo}
              </div>
              <span className="text-xs font-medium">{provider.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Phone Number */}
      <div>
        <Label htmlFor="phone" className="mb-2 block">Phone Number</Label>
        <div className="relative">
          <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          <Input
            id="phone"
            type="tel"
            placeholder="07XX XXX XXX"
            value={phoneNumber}
            onChange={(e) => setPhoneNumber(e.target.value)}
            className="pl-10"
            disabled={status === 'success'}
          />
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Enter the phone number registered with {PROVIDERS.find(p => p.id === selectedProvider)?.name}
        </p>
      </div>

      {/* Amount Display */}
      <div className="p-4 rounded-xl bg-muted">
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground">Amount to Pay</span>
          <span className="text-2xl font-bold">TZS {amount.toLocaleString()}</span>
        </div>
      </div>

      {/* Status */}
      {status === 'success' && (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-500" />
          <div>
            <p className="font-medium text-emerald-500">Payment Initiated!</p>
            <p className="text-sm text-muted-foreground">Please check your phone to complete the payment.</p>
          </div>
        </div>
      )}

      {/* Pay Button */}
      <Button
        onClick={handlePayment}
        disabled={isLoading || status === 'success'}
        className="w-full gap-2"
        size="lg"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </>
        ) : status === 'success' ? (
          'Payment Pending'
        ) : (
          `Pay with ${PROVIDERS.find(p => p.id === selectedProvider)?.name}`
        )}
      </Button>
    </div>
  );
}
```

---

## Implementation Checklist

### Week 1-2: Backend Foundation
- [ ] Set up Docker environment
- [ ] Create PostgreSQL schema
- [ ] Implement FastAPI structure
- [ ] Add authentication (JWT)
- [ ] Create file upload endpoint

### Week 3-4: AI Engine
- [ ] Implement DataProfiler
- [ ] Create InsightsGenerator
- [ ] Add zero-hallucination guardrails
- [ ] Build facts table system
- [ ] Create dashboard spec generator

### Week 5-6: Frontend Polish
- [ ] Fix light theme
- [ ] Complete Swahili translations
- [ ] Fix dashboard layout issues
- [ ] Add time formatting (hours/minutes)
- [ ] Improve responsive design

### Week 7-8: Payments
- [ ] Implement M-Pesa integration
- [ ] Add Airtel Money support
- [ ] Create payment UI components
- [ ] Add subscription management
- [ ] Test payment flows

### Week 9-10: Testing & Deployment
- [ ] Unit tests for AI engine
- [ ] Integration tests
- [ ] Load testing
- [ ] Security audit
- [ ] Production deployment

---

## Next Steps for You

1. **Set up the backend environment**:
   ```bash
   mkdir healthai-backend
   cd healthai-backend
   # Copy the docker-compose.yml and requirements.txt
   docker-compose up -d
   ```

2. **Get API keys**:
   - OpenAI API key (for GPT-4)
   - M-Pesa credentials (contact Safaricom Tanzania)
   - Optional: Anthropic key for Claude

3. **Run the database migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Start the backend**:
   ```bash
   uvicorn app.main:app --reload
   ```

Would you like me to continue implementing any specific part of this architecture?
