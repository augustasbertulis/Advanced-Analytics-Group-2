"""
@author:
@since: 19th August 2025
@version:
@source:
@code:
@param:
"""

from src import categories
from src import descriptions
from src import games
from src import genres
from src import promotional
from src import reviews
from src import steamspy_insights
from src import tags
from src import csv_merge
from src import further_preprocess

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
