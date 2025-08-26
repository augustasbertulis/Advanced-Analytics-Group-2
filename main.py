"""
@author:
@since: 19th August 2025
@version:
@source:
@code:
@param:
"""

from src.preprocessing import categories, descriptions, reviews, promotional, steamspy_insights, tags, genres, games, \
    further_preprocess, csv_merge
from src.processing import risk_kpi_calculator, model, get_attribute_table, plot_attribute_table, sensitivity_analysis, markowitz_model

if __name__ == "__main__":
    categories.main()
    descriptions.main()
    games.main()
    genres.main()
    promotional.main()
    reviews.main()
    steamspy_insights.main()
    tags.main()
    csv_merge.main()
    further_preprocess.main()
    risk_kpi_calculator.main()
    model.main()
    markowitz_model.run_pseudo_markowitz()

    # added analytics
    sensitivity_analysis.run_model_weight_sensitivity()
    get_attribute_table.main()
    plot_attribute_table.main()