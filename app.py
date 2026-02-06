from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import csv
import glob
import json

app = Flask(__name__, static_folder='static')
CORS(app)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')

def parse_traffic_state_pred_csv(filepath):
    """Parse traffic_state_pred CSV file and extract step 3,6,9,12 and average"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header: MAE, RMSE, MAPE
        for row in reader:
            if row and len(row) >= 6:
                try:
                    mae = float(row[4]) if row[4] != 'inf' else float('inf')
                    mape = float(row[5]) if row[5] != 'inf' else float('inf')
                    rmse = float(row[7]) if row[7] != 'inf' else float('inf')
                    data.append({'mae': mae, 'rmse': rmse, 'mape': mape})
                except (ValueError, IndexError):
                    continue

    result = {}
    # Get specific steps (1-indexed in display, 0-indexed in array)
    for step in [3, 6, 9, 12]:
        if len(data) >= step:
            result[f'step_{step}'] = data[step - 1]

    # Calculate average of all steps
    if data:
        avg_mae = sum(d['mae'] for d in data if d['mae'] != float('inf')) / len(data)
        avg_rmse = sum(d['rmse'] for d in data if d['rmse'] != float('inf')) / len(data)
        avg_mape = sum(d['mape'] for d in data if d['mape'] != float('inf')) / len(data) if any(d['mape'] != float('inf') for d in data) else float('inf')
        result['average'] = {'mae': avg_mae, 'rmse': avg_rmse, 'mape': avg_mape}

    return result


def parse_eta_csv(filepath):
    """Parse ETA CSV file and extract masked_MAE, masked_MAPE, masked_RMSE"""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers_lower = [h.lower() for h in headers]

        for row in reader:
            if row:
                result = {}
                for i, header in enumerate(headers_lower):
                    if header in ['masked_mae', 'masked_mape', 'masked_rmse']:
                        try:
                            result[header] = float(row[i]) if row[i] != 'inf' else float('inf')
                        except (ValueError, IndexError):
                            result[header] = None
                return result
    return {}


def parse_traj_loc_pred_json(filepath):
    """Parse traj_loc_pred JSON file and extract ACC@1, ACC@5, ACC@10, ACC@20, MRR@20, NDCG@20"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    result = {
        'acc_1': data.get('ACC@1'),
        'acc_5': data.get('ACC@5'),
        'acc_10': data.get('ACC@10'),
        'acc_20': data.get('ACC@20'),
        'mrr_20': data.get('MRR@20'),
        'ndcg_20': data.get('NDCG@20')
    }
    return result


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/tasks')
def get_tasks():
    """Get list of all tasks"""
    tasks = []
    if os.path.exists(RESULT_DIR):
        for task in os.listdir(RESULT_DIR):
            task_path = os.path.join(RESULT_DIR, task)
            if os.path.isdir(task_path):
                tasks.append(task)
    return jsonify(tasks)


@app.route('/api/tasks/<task>/datasets')
def get_datasets(task):
    """Get list of datasets for a task"""
    datasets = []
    task_path = os.path.join(RESULT_DIR, task)
    if os.path.exists(task_path):
        for dataset in os.listdir(task_path):
            dataset_path = os.path.join(task_path, dataset)
            if os.path.isdir(dataset_path):
                datasets.append(dataset)
    return jsonify(datasets)


@app.route('/api/tasks/<task>/datasets/<dataset>/rankings')
def get_rankings(task, dataset):
    """Get rankings for a specific dataset"""
    dataset_path = os.path.join(RESULT_DIR, task, dataset)
    rankings = []

    if not os.path.exists(dataset_path):
        return jsonify([])

    if task == 'traj_loc_pred':
        json_files = glob.glob(os.path.join(dataset_path, '*.json'))
        for json_file in json_files:
            model_name = os.path.splitext(os.path.basename(json_file))[0]
            data = parse_traj_loc_pred_json(json_file)
            rankings.append({
                'model': model_name,
                'data': data
            })
    else:
        csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
        for csv_file in csv_files:
            model_name = os.path.splitext(os.path.basename(csv_file))[0]

            if task == 'traffic_state_pred':
                data = parse_traffic_state_pred_csv(csv_file)
                rankings.append({
                    'model': model_name,
                    'data': data
                })
            elif task == 'eta':
                data = parse_eta_csv(csv_file)
                rankings.append({
                    'model': model_name,
                    'data': data
                })

    return jsonify(rankings)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8085)
