from flask import Flask, jsonify, send_from_directory, request, Response, redirect
from flask_cors import CORS
import os
import csv
import glob
import json
import sys
import multiprocessing
import requests as req_lib

app = Flask(__name__, static_folder='static')
CORS(app)

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'result')
MIGRATION_JSON = os.path.join(os.path.dirname(__file__), 'migration_all.json')
AGENTCITY_DIR = os.path.join(os.path.dirname(__file__), 'AgentCity')
AGENTCITY_PORT = 8000
AGENTCITY_BASE = f'http://localhost:{AGENTCITY_PORT}'


def _run_uvicorn():
    """Entry point for the AgentCity backend process."""
    sys.path.insert(0, AGENTCITY_DIR)
    os.chdir(AGENTCITY_DIR)
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=AGENTCITY_PORT)


def start_agentcity():
    """Start AgentCity FastAPI backend in a new process if not already running."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', AGENTCITY_PORT)) == 0:
            return  # already running
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return  # only start in the reloader child process, not the monitor
    proc = multiprocessing.Process(target=_run_uvicorn, daemon=True)
    proc.start()

start_agentcity()

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


def parse_map_matching_csv(filepath):
    """Parse map_matching CSV file and extract RMF, AN, AL"""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        headers_map = {h.strip(): i for i, h in enumerate(headers)}
        for row in reader:
            if row:
                try:
                    return {
                        'rmf': float(row[headers_map['RMF']]),
                        'an': float(row[headers_map['AN']]),
                        'al': float(row[headers_map['AL']])
                    }
                except (ValueError, KeyError, IndexError):
                    continue
    return {}


def parse_map_matching_json(filepath):
    """Parse map_matching JSON file and extract best RMF (smallest), AN (largest), AL (largest) across all runs"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Skip GeoJSON files (they don't contain metrics)
    if data.get('type') == 'FeatureCollection':
        return None

    best_rmf = None
    best_an = None
    best_al = None
    details = data.get('details', {})
    for group in details.values():
        for run in group.values():
            rmf = run.get('RMF')
            an = run.get('AN')
            al = run.get('AL')
            # RMF: smaller is better
            if rmf is not None and (best_rmf is None or rmf < best_rmf):
                best_rmf = rmf
            # AN: larger is better
            if an is not None and (best_an is None or an > best_an):
                best_an = an
            # AL: larger is better
            if al is not None and (best_al is None or al > best_al):
                best_al = al

    return {
        'rmf': best_rmf,
        'an': best_an,
        'al': best_al
    }


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


@app.route('/api/model_links')
def get_model_links():
    """Get model_name -> pdf_link mapping from migration_all.json"""
    links = {}
    if os.path.exists(MIGRATION_JSON):
        with open(MIGRATION_JSON, 'r') as f:
            data = json.load(f)
        for entry in data:
            name = entry.get('model_name')
            pdf = entry.get('pdf_link')
            if name and pdf:
                links[name] = pdf
    return jsonify(links)


@app.route('/api/paper_search')
def paper_search():
    """Search papers in migration_all.json by keyword relevance."""
    keyword = request.args.get('q', '').lower().strip()
    if not keyword or not os.path.exists(MIGRATION_JSON):
        return jsonify([])
    with open(MIGRATION_JSON, 'r') as f:
        data = json.load(f)
    terms = keyword.split()

    def score(entry):
        text = ' '.join([
            entry.get('title', ''),
            entry.get('model_name', ''),
            entry.get('conference', ''),
            ' '.join(entry.get('datasets', [])),
        ]).lower()
        return sum(text.count(t) for t in terms)

    scored = [(score(e), e) for e in data if e.get('title') and e.get('pdf_link')]
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [
        {'title': e.get('title'), 'pdf_link': e.get('pdf_link')}
        for _, e in scored[:5]
    ]
    return jsonify(results)



def agentcity():
    return redirect('/AgentCity/')


@app.route('/AgentCity/')
def agentcity_index():
    return send_from_directory(os.path.join(AGENTCITY_DIR, 'frontend'), 'index.html')


@app.route('/AgentCity/static/<path:filename>')
@app.route('/AgentCity/<path:filename>')
def agentcity_static(filename):
    return send_from_directory(os.path.join(AGENTCITY_DIR, 'frontend'), filename)


HOP_BY_HOP_HEADERS = frozenset([
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailers', 'transfer-encoding', 'upgrade',
    'server', 'date',
])

@app.route('/AgentCity/api', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/AgentCity/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/api', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def agentcity_proxy(path):
    url = f'{AGENTCITY_BASE}/api/{path}'
    try:
        resp = req_lib.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            data=request.get_data(),
            params=request.args,
            timeout=60,
            allow_redirects=False,
        )
        filtered_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        return Response(resp.content, status=resp.status_code, headers=filtered_headers)
    except req_lib.exceptions.ConnectionError:
        return jsonify({'error': 'AgentCity backend unavailable'}), 503


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/quickstart')
def quickstart():
    return send_from_directory('static', 'quickstart.html')


@app.route('/leaderboard')
def leaderboard():
    return send_from_directory('static', 'leaderboard.html')


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

    if task == 'map_matching':
        # Collect all model names from both CSV and JSON files
        csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
        json_files = glob.glob(os.path.join(dataset_path, '*.json'))
        csv_models = {os.path.splitext(os.path.basename(f))[0]: f for f in csv_files}
        json_models = {os.path.splitext(os.path.basename(f))[0]: f for f in json_files}
        all_models = set(csv_models.keys()) | set(json_models.keys())

        for model_name in all_models:
            data = None
            if model_name in csv_models:
                # Prioritize CSV
                data = parse_map_matching_csv(csv_models[model_name])
            elif model_name in json_models:
                # Fall back to JSON
                data = parse_map_matching_json(json_models[model_name])
            if data:
                rankings.append({
                    'model': model_name,
                    'data': data
                })
    elif task == 'traj_loc_pred':
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
    app.run(debug=True, host='0.0.0.0', port=8001)
