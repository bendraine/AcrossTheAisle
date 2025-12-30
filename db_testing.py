from document_store import get_store_stats
import json

stats = get_store_stats()
print(json.dumps(stats, indent=2))