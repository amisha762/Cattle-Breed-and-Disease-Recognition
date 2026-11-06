import os
import io
import torch
import torch.nn as nn
import gc

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')



import logging
logging.basicConfig(level=logging.INFO)

from flask import session, redirect
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash






BASE_DIR = os.path.dirname(__file__)


_MODEL_CANDIDATES = [
    {
        'breed': os.path.join(BASE_DIR, 'resnet18_cattle.pth'),
        'disease': os.path.join(BASE_DIR, 'disease_model.pth'),
        'disease_alt': os.path.join(BASE_DIR, 'cow_disease_model_full.pth'),
    },
    {
        'breed': os.path.join(os.path.dirname(BASE_DIR), 'resnet18_cattle.pth'),
        'disease': os.path.join(os.path.dirname(BASE_DIR), 'disease_model.pth'),
    },
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _find_existing_models():
    for cand in _MODEL_CANDIDATES:
        breed_ok = os.path.exists(cand['breed'])

        disease_path = cand.get('disease')
        alt_path = cand.get('disease_alt')

        # Accept if either the explicit disease file exists or the alt file exists.
        disease_ok = bool(disease_path) and os.path.exists(disease_path)
        alt_ok = bool(alt_path) and os.path.exists(alt_path)

        if breed_ok and (disease_ok or alt_ok):
            # Ensure returned dict has 'disease' filled with the actual path
            # so the rest of the code can keep using _models['disease'].
            if not disease_ok and alt_ok:
                cand = dict(cand)
                cand['disease'] = alt_path
            return cand
    return None


def _missing_models_error_json():
    paths = [
        {'breed_model': c['breed'], 'disease_model': c['disease']}
        for c in _MODEL_CANDIDATES
    ]
    return {
        'success': False,
        'error': 'Model files not found. Expected breed_model.pth and disease_model.pth in either app folder or its parent directory.',
        'searched_paths': paths,
        'suggestion': 'Place breed_model.pth & disease_model.pth in c:/cattle_app/ (preferred) or in the parent directory of app.py.'
    }


_models = _find_existing_models()
if not _models:
    breed_model = None
    disease_model = None
    BREED_CLASSES = []
    DISEASE_CLASSES = []
    _MODELS_ERROR = _missing_models_error_json()
    logging.error("%s", _MODELS_ERROR)
else:
    BREED_MODEL_PATH = _models['breed']
    # Use explicit disease model if present, otherwise fall back to alt checkpoint.
    DISEASE_MODEL_PATH = _models.get('disease') if _models.get('disease') else _models.get('disease_alt')
    _MODELS_ERROR = None
    logging.info("Loading models from:\n  breed=%s\n  disease=%s", BREED_MODEL_PATH, DISEASE_MODEL_PATH)


   
    def _load_checkpoint_generic(ckpt_obj, *, kind: str):
        """Supports two common .pth formats:
        1) dict with keys: class_names + model_state_dict
        2) raw state_dict (weights only)
        """
        if isinstance(ckpt_obj, dict):
            # format A: full checkpoint
            if 'model_state_dict' in ckpt_obj:
                weights = ckpt_obj['model_state_dict']
                class_names = ckpt_obj.get('class_names')
                return class_names, weights

         
            looks_like_state_dict = any(
                isinstance(v, torch.Tensor) for v in ckpt_obj.values()
            ) or any(
                isinstance(k, str) and (k.startswith('module.') or '.' in k) for k in ckpt_obj.keys()
            )
            if looks_like_state_dict:
                return None, ckpt_obj

        raise KeyError(f"{kind} checkpoint format not recognized. Type={type(ckpt_obj)}")

    
    breed_ckpt_obj = torch.load(BREED_MODEL_PATH, map_location=device)
    breed_class_names, breed_state_dict = _load_checkpoint_generic(
        breed_ckpt_obj, kind='Breed'
    )

  
    def _infer_num_classes_from_state_dict(state_dict, *, model_kind: str):
        # Try common fc naming
        candidate_keys = [
            'fc.weight', 'fc.bias',
            'model.fc.weight', 'model.fc.bias',
            'module.fc.weight', 'module.fc.bias',
        ]
        for ck in candidate_keys:
            if ck in state_dict and hasattr(state_dict[ck], 'shape'):
                w = state_dict[ck]
                # fc.weight: [out_features, in_features]
                if getattr(w, 'ndim', None) == 2:
                    return int(w.shape[0])
                if getattr(w, 'ndim', None) == 1:
                    return int(w.shape[0])

    
        linear_weight_keys = [k for k in state_dict.keys() if k.endswith('fc.weight')]
        if linear_weight_keys:
            w = state_dict[linear_weight_keys[0]]
            if getattr(w, 'ndim', None) == 2:
                return int(w.shape[0])

        raise KeyError(f"Could not infer num classes for {model_kind} from provided state_dict keys")

    breed_num_classes = _infer_num_classes_from_state_dict(breed_state_dict, model_kind='Breed')

    if not breed_class_names:
        
        BREED_CLASSES = [f"breed_{i}" for i in range(breed_num_classes)]
        logging.warning("Breed checkpoint missing 'class_names'. Using placeholders with %d classes.", breed_num_classes)
    else:
        BREED_CLASSES = breed_class_names
        
        if len(BREED_CLASSES) != breed_num_classes:
            logging.warning("Breed class_names length (%d) != inferred num_classes (%d). Using inferred.", len(BREED_CLASSES), breed_num_classes)
            BREED_CLASSES = [f"breed_{i}" for i in range(breed_num_classes)]

    breed_model = models.resnet18(pretrained=False)
    breed_model.fc = nn.Linear(breed_model.fc.in_features, len(BREED_CLASSES))
    # strict=False to tolerate prefix mismatches (e.g., 'module.')
    breed_model.load_state_dict(breed_state_dict, strict=False)
    breed_model.to(device).eval()

    del breed_ckpt_obj, breed_state_dict
    gc.collect()

  
    disease_ckpt_obj = torch.load(DISEASE_MODEL_PATH, map_location=device)
    disease_class_names, disease_state_dict = _load_checkpoint_generic(
        disease_ckpt_obj, kind='Disease'
    )

    disease_num_classes = _infer_num_classes_from_state_dict(disease_state_dict, model_kind='Disease')

    if not disease_class_names:
        DISEASE_CLASSES = [f"disease_{i}" for i in range(disease_num_classes)]
        logging.warning("Disease checkpoint missing 'class_names'. Using placeholders with %d classes.", disease_num_classes)
    else:
        DISEASE_CLASSES = disease_class_names
        if len(DISEASE_CLASSES) != disease_num_classes:
            logging.warning("Disease class_names length (%d) != inferred num_classes (%d). Using inferred.", len(DISEASE_CLASSES), disease_num_classes)
            DISEASE_CLASSES = [f"disease_{i}" for i in range(disease_num_classes)]

    # Debug: log actual disease class names stored in checkpoint
    logging.info("Disease checkpoint class_names (%s): %s", len(DISEASE_CLASSES), DISEASE_CLASSES)

    disease_model = models.resnet18(pretrained=False)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, len(DISEASE_CLASSES))
    disease_model.load_state_dict(disease_state_dict, strict=False)
    disease_model.to(device).eval()

    del disease_ckpt_obj, disease_state_dict
    gc.collect()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DISEASE_META = {
    
    'foot-and-mouth': {
        'display': 'Foot-and-Mouth Disease',
        'severity': 'high',
        'description': 'Highly contagious viral disease. Isolate immediately and contact a veterinarian.',
        'icon': 'high'
    },
    'lumpy': {
        'display': 'Lumpy Skin Disease',
        'severity': 'medium',
        'description': 'Viral disease causing skin nodules. Consult a vet for vaccination and treatment.',
        'icon': 'medium'
    },
    'healthy': {
        'display': 'Healthy',
        'severity': 'none',
        'description': 'No visible signs of disease detected. Animal appears healthy.',
        'icon': 'none'
    }
}

DISEASE_ALIASES = {
    'lumpy_skin': 'lumpy',
    'lumpy-skin': 'lumpy',
    'lumpy skin': 'lumpy',
    'lumpy': 'lumpy',

    # Foot-and-Mouth Disease aliases
    'foot_and_mouth': 'foot-and-mouth',
    'foot-and-mouth': 'foot-and-mouth',
    'foot to mouth': 'foot-and-mouth',
    'foot-to-mouth': 'foot-and-mouth',
    'fmd': 'foot-and-mouth',
    'ftm': 'foot-and-mouth',

    # Healthy aliases
    'healthy': 'healthy',
    'no_disease': 'healthy',
    'no disease': 'healthy',
}


def normalize_disease_key(key: str) -> str:
    if key is None:
        return ''
    k = str(key).strip().lower()
    # direct canonical match first
    if k in DISEASE_META:
        return k
    # alias normalization
    if k in DISEASE_ALIASES:
        return DISEASE_ALIASES[k]
    # fallback: try a few heuristics
    k2 = k.replace('_', '-').replace(' ', '-')
    if k2 in DISEASE_META:
        return k2
    if 'lumpy' in k2:
        return 'lumpy'
    if ('foot' in k2 and 'mouth' in k2) or ('fmd' in k2):
        return 'foot-and-mouth'
    if 'healthy' in k2 or 'normal' in k2:
        return 'healthy'
    return k


def fmt(name):
    return name.replace('_', ' ').title()

def load_image(file):
    img_bytes = file.read()
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

def run_model(model, img):
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, min(5, len(probs)))

    return top_probs.tolist(), top_idx.tolist()




@app.route('/predict/breed', methods=['POST'])
def predict_breed():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    if breed_model is None:
        return jsonify(_MODELS_ERROR), 500
    try:
        img = load_image(request.files['image'])
        probs, idxs = run_model(breed_model, img)
        predictions = [
            {
                'breed': BREED_CLASSES[i],
                'breed_display': fmt(BREED_CLASSES[i]),
                'confidence': round(p * 100, 2)
            }
            for p, i in zip(probs, idxs)
        ]
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_breed': predictions[0]['breed_display'],
            'top_confidence': predictions[0]['confidence'],
            # for frontend dashboards
            'mode': 'breed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

TREATMENTS = {
    'healthy': [
        'Animal appears healthy. Continue balanced nutrition.',
        'Maintain regular vaccination and hygiene.',
        'Schedule routine veterinary checkups.'
    ],
    'lumpy': [
        'Isolate the infected animal to prevent spread.',
        'Control flies/mosquitoes and keep the environment clean.',
        'Clean skin nodules with antiseptic; consult a veterinarian for specific treatment.',
        'Provide nutritious feed and plenty of water.'
    ],
    'foot-and-mouth': [
        'Isolate the animal from the herd immediately.',
        'Provide soft, easily digestible feed and adequate hydration.',
        'Clean mouth/foot lesions with antiseptic and keep lesions dry.',
        'Contact a veterinarian immediately for supportive/antiviral care.'
    ],
}


@app.route('/predict/disease', methods=['POST'])
def predict_disease():

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    if disease_model is None:
        return jsonify(_MODELS_ERROR), 500
    try:
        img = load_image(request.files['image'])
        probs, idxs = run_model(disease_model, img)
        predictions = []
        for p, i in zip(probs, idxs):
            raw_key = DISEASE_CLASSES[i]
            norm_key = normalize_disease_key(raw_key)
            meta = DISEASE_META.get(norm_key, {
                'display': fmt(raw_key),
                'severity': 'unknown',
                'description': '',
                'icon': 'unknown'
            })
            predictions.append({
                'key': raw_key,
                'normalized_key': norm_key,
                'display': meta['display'],
                'confidence': round(p * 100, 2),
                'severity': meta['severity'],
                'description': meta['description'],
                'icon': meta['icon'],
                'treatment': TREATMENTS.get(norm_key, [])
            })

        top = predictions[0]
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_disease': top['display'],
            'top_confidence': top['confidence'],
            'severity': top['severity'],
            'description': top['description'],
            'treatment': top.get('treatment', []),
            'mode': 'disease'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/both', methods=['POST'])
def predict_both():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    if breed_model is None or disease_model is None:
        return jsonify(_MODELS_ERROR), 500
    try:
        img_bytes = request.files['image'].read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        b_probs, b_idxs = run_model(breed_model, img)
        breed_preds = [{'breed': BREED_CLASSES[i],
                        'breed_display': fmt(BREED_CLASSES[i]),
                        'confidence': round(p * 100, 2)}
                       for p, i in zip(b_probs, b_idxs)]

        d_probs, d_idxs = run_model(disease_model, img)
        disease_preds = []
        for p, i in zip(d_probs, d_idxs):
            raw_key = DISEASE_CLASSES[i]
            norm_key = normalize_disease_key(raw_key)
            meta = DISEASE_META.get(norm_key, {
                'display': fmt(raw_key),
                'severity': 'unknown',
                'description': '',
                'icon': 'unknown'
            })
            disease_preds.append({
                'key': raw_key,
                'normalized_key': norm_key,
                'display': meta['display'],
                'confidence': round(p * 100, 2),
                'severity': meta['severity'],
                'description': meta['description']
            })

        return jsonify({
            'success': True,
            'breed': {'predictions': breed_preds,
                      'top_breed': breed_preds[0]['breed_display'],
                      'top_confidence': breed_preds[0]['confidence'],
                      'mode': 'breed'},
            'disease': {'predictions': disease_preds,
                        'top_disease': disease_preds[0]['display'],
                        'top_confidence': disease_preds[0]['confidence'],
                        'severity': disease_preds[0]['severity'],
                        'description': disease_preds[0]['description'],
                        'treatment': disease_preds[0].get('treatment', []),
                        'mode': 'disease'}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/breeds')
def breeds():

    return jsonify({'breeds': [fmt(b) for b in BREED_CLASSES], 'count': len(BREED_CLASSES)})

@app.route('/diseases')
def diseases():
    return jsonify({'diseases': DISEASE_CLASSES, 'count': len(DISEASE_CLASSES)})


USERS = {

    'admin': generate_password_hash('1234'),
}



def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return redirect('/login')
        return fn(*args, **kwargs)
    return wrapper


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', error=None)

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''

    stored = USERS.get(username)
    if not stored or not check_password_hash(stored, password):
        return render_template('login.html', error='Invalid username or password')

    session['user'] = username
    return redirect('/frontend')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


@app.route('/frontend')
@login_required
def frontend():
    # Main single-page UI
    return render_template('login.html')






@app.route('/')

def index():
    # keep landing simple: go to login or frontend
    if session.get('user'):
        return redirect('/frontend')
    return redirect('/login')


if __name__ == '__main__':

    if breed_model is None or disease_model is None:
        print("Models missing. Server will run but prediction endpoints will return an error JSON.")
        print(f"Searched paths: {_MODELS_ERROR.get('searched_paths') if _MODELS_ERROR else []}")
    else:
        print(f"Breed model  : {len(BREED_CLASSES)} classes")
        print(f"Disease model: {len(DISEASE_CLASSES)} classes")

    app.run(debug=True, port=5000)
