import pandas as pd
from evaluation_dataset import GenreCategory

def analyze_dataset():
    # Load data
    try:
        df = pd.read_csv('data/combined_preprocessed.csv')
        print(f"Total books in DB: {len(df)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 1. Genre Analysis
    print("=== Genre Distribution (Top 20) ===")
    genre_col = '대표분류(대분류명)'
    sub_genre_col = '구분'
    
    if genre_col in df.columns:
        print(f"\n[Column: {genre_col}]")
        print(df[genre_col].value_counts().head(20))
    
    if sub_genre_col in df.columns:
        print(f"\n[Column: {sub_genre_col}]")
        print(df[sub_genre_col].value_counts().head(20))

    # Combine both for evaluation check
    print("\n=== Evaluation Dataset Genre Check (checking both columns) ===")
    
    # Get all unique values from both columns
    all_genres = set()
    if genre_col in df.columns:
        all_genres.update(df[genre_col].dropna().unique())
    if sub_genre_col in df.columns:
        all_genres.update(df[sub_genre_col].dropna().unique())
        
    eval_genres = [g.value for g in GenreCategory]
    missing_genres = []
    
    for g in eval_genres:
        if g not in all_genres:
             missing_genres.append(g)
        else:
             count1 = df[genre_col].value_counts().get(g, 0) if genre_col in df.columns else 0
             count2 = df[sub_genre_col].value_counts().get(g, 0) if sub_genre_col in df.columns else 0
             print(f"✓ {g}: {count1 + count2} occurrences (Total)")
    
    if missing_genres:
        print(f"\n⚠️ Missing Genres in CSV: {missing_genres}")

    # 2. Theme/Keyword Analysis
    print("\n=== Keyword Coverage Check ===")
    keywords = [
        "SF", "마케팅", "한국소설", "역사", "에세이", 
        "우울", "위로", "동기부여", "여행", 
        "군대", "취업", "미스터리", "추리"
    ]
    
    # Simple keyword search in title and description
    df['text_search'] = df['상품명'].fillna('') + " " + df['책소개'].fillna('')
    
    for kw in keywords:
        count = df['text_search'].str.contains(kw, case=False).sum()
        print(f"Keyword '{kw}': {count} books")

if __name__ == "__main__":
    analyze_dataset()
