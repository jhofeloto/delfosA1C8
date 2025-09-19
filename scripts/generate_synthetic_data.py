#!/usr/bin/env python
"""Generate synthetic diabetes dataset"""
import numpy as np
import pandas as pd
import random

def generate_synthetic_data(n=100, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    for i in range(n):
        rand = np.random.random()
        if rand < 0.40:
            cat = 0
            glucose = np.random.uniform(70, 99)
        elif rand < 0.75:
            cat = 1
            glucose = np.random.uniform(100, 126)
        else:
            cat = 2
            glucose = np.random.uniform(127, 200)
        
        data.append({
            'edad': 30 + cat*10 + np.random.normal(0, 15),
            'talla': np.random.normal(165, 10),
            'peso': 60 + cat*10 + np.random.normal(0, 15),
            'imc': 22 + cat*4 + np.random.normal(0, 3),
            'tas': 110 + cat*15 + np.random.normal(0, 10),
            'tad': 70 + cat*10 + np.random.normal(0, 7),
            'perimetro_abdominal': 80 + cat*10 + np.random.normal(0, 10),
            'puntaje_total': 5 + cat*5 + np.random.normal(0, 2),
            'riesgo_dm': cat*0.35 + np.random.random()*0.3,
            'Resultado': glucose + np.random.normal(0, 5)
        })
    
    df = pd.DataFrame(data)
    df['peso'] = df['imc'] * (df['talla']/100)**2
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(2)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(100)
    df.to_csv("data/raw/outputglucosa.csv", index=False)
    print(f"✅ Generated {len(df)} records")
    print(f"   Normal: {(df['Resultado'] < 100).sum()}")
    print(f"   Prediabetes: {((df['Resultado'] >= 100) & (df['Resultado'] <= 126)).sum()}")
    print(f"   Diabetes: {(df['Resultado'] > 126).sum()}")
