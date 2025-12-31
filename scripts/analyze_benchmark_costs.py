import csv
import locale

# Set locale for currency formatting if needed, though simple f-strings are fine
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def analyze_costs(csv_path):
    models = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row['Date']
            # Filter for benchmark dates
            if not date.startswith('2025-12-30') and not date.startswith('2025-12-31'):
                continue
                
            model = row['Slug']
            
            # Skip free models
            if ':free' in model:
                continue
            
            # Parse numbers
            try:
                usage_cost = float(row['Usage'])
                requests = int(row['Requests'])
                # Some rows might be 0 requests if it was just a ping?
                if requests == 0:
                    continue
            except ValueError:
                continue

            if model not in models:
                models[model] = {'cost': 0.0, 'requests': 0}
            
            models[model]['cost'] += usage_cost
            models[model]['requests'] += requests

    # Calculate per-page stats and projections
    results = []
    total_pages_project = 1_220_000
    
    print(f"{'Model':<45} | {'Reqs':<5} | {'Cost/Page':<10} | {'1.22M Cost':<12}")
    print("-" * 80)
    
    for model, data in models.items():
        if data['requests'] < 5: # Skip tiny tests or interrupted runs
            continue
            
        avg_cost = data['cost'] / data['requests']
        projected_total = avg_cost * total_pages_project
        
        results.append({
            'model': model,
            'requests': data['requests'],
            'avg_cost': avg_cost,
            'projected': projected_total
        })

    # Sort by projected cost (cheapest first)
    results.sort(key=lambda x: x['projected'])
    
    for r in results:
        print(f"{r['model']:<45} | {r['requests']:<5} | ${r['avg_cost']:.6f}  | ${r['projected']:,.2f}")

if __name__ == "__main__":
    analyze_costs('openrouter-activity-20251231-003046.csv')
