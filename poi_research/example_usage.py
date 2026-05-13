from __future__ import annotations

from poi_research import process_pois


EXAMPLE_INPUT = [
    {
        "poi_id": "beach_1",
        "place_name": "Windansea Beach",
        "category": "Beach",
        "city": "La Jolla",
        "region": "California",
    },
    {
        "poi_id": "museum_1",
        "place_name": "The Getty Center",
        "category": "Museum",
        "city": "Los Angeles",
        "region": "California",
    },
]


def main() -> None:
    results = process_pois(EXAMPLE_INPUT, max_workers=2, debug=True)
    for item in results:
        print(item.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
