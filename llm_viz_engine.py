# File: llm_viz_engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from jinja2 import Template
from groq import Groq

load_dotenv()

class LLMVisualizationEngine:
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not provided.")

        self.client = Groq(api_key=self.groq_api_key)
        self.df = None
        self.data_profile = {}
        self.insights = []
        self.visualizations = []
        self.data_context = ""
        self.chart_counter = 0
        self.chart_paths = []

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_data(self, file_path: str) -> pd.DataFrame:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for enc in encodings:
            try:
                self.df = pd.read_csv(file_path, encoding=enc)
                self.df.columns = self.df.columns.astype(str).str.strip()
                print(f"✅ Loaded data using encoding: {enc}")
                return self.df
            except Exception as e:
                print(f"⚠️ Failed with encoding {enc}: {e}")
        raise ValueError("❌ Could not load CSV. Try saving it as UTF-8.")

    def create_derived_features(self):
        if self.df is None:
            raise ValueError("No data loaded.")
        try:
            dob_cols = [col for col in self.df.columns if 'birth' in col.lower() or 'dob' in col.lower()]
            for col in dob_cols:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                self.df['age'] = self.df[col].apply(lambda x: datetime.now().year - x.year if pd.notnull(x) else None)
                print("✅ Derived 'age' column")
        except Exception as e:
            print(f"⚠️ Failed to create derived features: {e}")

    def profile_data(self) -> Dict[str, Any]:
        profile = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': [],
            'unique_counts': {},
            'summary_stats': {},
            'correlations': {},
            'outliers': {},
            'skewness': {}
        }
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].dropna().iloc[:100])
                    profile['datetime_columns'].append(col)
                except: pass
            profile['unique_counts'][col] = self.df[col].nunique()

        if profile['numeric_columns']:
            profile['summary_stats'] = self.df[profile['numeric_columns']].describe().to_dict()
            profile['correlations'] = self.df[profile['numeric_columns']].corr().to_dict()
            for col in profile['numeric_columns']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
                profile['outliers'][col] = len(outliers)
                profile['skewness'][col] = self.df[col].skew()

        self.data_profile = profile
        return profile

    def detect_data_context(self) -> str:
        column_names = ', '.join(self.df.columns.tolist())
        sample_data = self.df.head(3).to_string()
        prompt = f"""
        Analyze this dataset. Here are the columns: {column_names}
        Sample rows:
        {sample_data}
        The engine can derive new features like 'age' from 'DOB'. Suggest a relevant domain.
        """
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            self.data_context = response.choices[0].message.content.strip()
            return self.data_context
        except:
            self.data_context = "General dataset"
            return self.data_context

    def generate_insights(self) -> List[str]:
        profile_summary = f"""
        Shape: {self.data_profile['shape']}
        Numeric: {self.data_profile['numeric_columns']}
        Categorical: {self.data_profile['categorical_columns']}
        Missing: {dict([(k, v) for k, v in self.data_profile['missing_values'].items() if v > 0])}
        Correlations: {dict(list(self.data_profile.get('correlations', {}).items())[:5])}
        Outliers: {dict([(k, v) for k, v in self.data_profile.get('outliers', {}).items() if v > 0])}
        Context: {self.data_context}
        """
        prompt = f"""
        Provide 5-7 insights for this dataset:
        {profile_summary}
        """
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            text = response.choices[0].message.content.strip()
            self.insights = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith("#")]
            return self.insights
        except:
            self.insights = ["Insight generation failed"]
            return self.insights

    def suggest_visualizations(self) -> List[Dict[str, Any]]:
        suggestions = []
        for col in self.data_profile['numeric_columns']:
            suggestions.append({
                'type': 'histogram',
                'columns': [col],
                'title': f'Distribution of {col}',
                'description': f'Histogram of {col}'
            })
        if len(self.data_profile['numeric_columns']) > 2:
            suggestions.append({
                'type': 'correlation_heatmap',
                'columns': self.data_profile['numeric_columns'],
                'title': 'Correlation Matrix',
                'description': 'Heatmap of correlations'
            })
        return suggestions

    def create_visualization(self, cfg: Dict[str, Any]) -> str:
        self.chart_counter += 1
        chart_filename = f"chart_{self.chart_counter}.png"
        plt.figure(figsize=(10, 6))
        try:
            if cfg['type'] == 'histogram':
                plt.hist(self.df[cfg['columns'][0]].dropna(), bins=30, edgecolor='black')
            elif cfg['type'] == 'correlation_heatmap':
                sns.heatmap(self.df[cfg['columns']].corr(), annot=True, cmap='coolwarm')
            elif cfg['type'] == 'scatter_plot':
                x, y = cfg['columns'][0], cfg['columns'][1]
                plt.scatter(self.df[x], self.df[y], alpha=0.6)
                plt.xlabel(x)
                plt.ylabel(y)
            elif cfg['type'] == 'bar_chart':
                col = cfg['columns'][0]
                counts = self.df[col].value_counts().head(10)
                counts.plot(kind='bar')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            elif cfg['type'] == 'line_chart':
                x, y = cfg['columns'][0], cfg['columns'][1]
                plt.plot(self.df[x], self.df[y], marker='o')
                plt.xlabel(x)
                plt.ylabel(y)
            elif cfg['type'] == 'pie_chart':
                data = self.df[cfg['columns'][0]].value_counts().head(5)
                plt.pie(data, labels=data.index, autopct='%1.1f%%')
                plt.axis('equal')
            plt.title(cfg['title'])
            plt.tight_layout()
            plt.savefig(chart_filename, dpi=200)
            plt.close()
            self.chart_paths.append(chart_filename)
            return chart_filename
        except Exception as e:
            print(f"❌ Error generating chart: {e}")
            plt.close()
            return None

    def visualize_from_prompt(self, prompt: str) -> Optional[str]:
        try:
            col_list = ', '.join(self.df.columns)
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a chart generator. Output a JSON with type, columns, title, description."},
                    {"role": "user", "content": f"Available columns: {col_list}. Prompt: {prompt}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            content = response.choices[0].message.content
            match = re.search(r'{.*}', content, re.DOTALL)
            config = json.loads(match.group()) if match else None
            if config:
                return self.create_visualization(config)
        except Exception as e:
            print(f"❌ Prompt visualization failed: {e}")
        return None

    def visualize_multiple_from_prompt(self, prompt: str) -> List[str]:
        try:
            col_list = ', '.join(self.df.columns)
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "Output a list of chart configs as JSON."},
                    {"role": "user", "content": f"Columns: {col_list}. Prompt: {prompt}"}
                ],
                temperature=0.4,
                max_tokens=700
            )
            content = response.choices[0].message.content
            match = re.search(r'\[.*\]', content, re.DOTALL)
            configs = json.loads(match.group()) if match else []
            return [self.create_visualization(cfg) for cfg in configs if isinstance(cfg, dict)]
        except Exception as e:
            print(f"❌ Multiple viz error: {e}")
            return []

    def generate_html_report(self) -> str:
        html_template = """
        <html><body style='font-family:sans-serif;'>
        <h1 style='color:#007acc;'>📊 LLM Data Report</h1>
        <p><b>Context:</b> {{ data_context }}</p>
        <h2>🔍 Insights</h2>
        <ul>{% for i in insights %}<li>{{ i }}</li>{% endfor %}</ul>
        <h2>📈 Charts</h2>
        {% for viz in visualizations %}
        <div style='margin-bottom:30px;'>
        <h3>{{ viz.title }}</h3>
        <img src="data:image/png;base64,{{ viz.image_base64 }}" style="width:80%;">
        </div>
        {% endfor %}
        </body></html>
        """
        viz_data = []
        for i, path in enumerate(self.chart_paths):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    viz_data.append({"title": f"Chart {i+1}", "image_base64": b64})
        return Template(html_template).render(data_context=self.data_context, insights=self.insights, visualizations=viz_data)
