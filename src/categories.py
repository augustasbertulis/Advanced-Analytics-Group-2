import pandas as pd


CATEGORY_TRANSLATIONS = {
    "Family Sharing": ["Family Sharing", "Familienbibliothek", "Семейный доступ", "Partage familial", "Compartilhamento em família"],
    "Multi-player": ["Multi-player", "Mehrspieler", "Для нескольких игроков", "Multijoueur", "Multijogador"],
    "Co-op": ["Co-op", "Кооператив", "Coopération", "Cooperação"],
    "Steam Cloud": ["Steam Cloud", "Облако Steam", "Cloud Steam"],
    "Steam Achievements": ["Steam Achievements", "Достижения Steam", "Succès Steam", "Conquistas Steam"],
    "Remote Play Together": ["Remote Play Together", "Совместная игра удаленно", "Remote Play sur tablette", "Remote Play na TV"],
    "Valve Anti-Cheat enabled": ["Valve Anti-Cheat enabled", "Включён античит Valve", "Valve Anti-Cheat integriert"],
    "Stats": ["Stats", "Статистика", "Statistiques"],
}


    df["category"] = df["category"].apply(normalize_category)







def main():

    df_transformed = process_data(df)

    df_transformed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] written {len(df_transformed)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
