# Cattle Breed Identifier

AI-powered cattle breed classification web app using ResNet-18.

## 🚀 Live Demo

The Cattle Breed and Disease Recognition System is deployed on Render.

[Click Here to Open the Live Application](https://cattle-breed-disease-recognition.onrender.com)

## Features
- Identifies 73 cattle breeds (Indian indigenous, buffalo, international dairy)
- Returns top-5 predictions with confidence scores
- Drag-and-drop image upload
- Clean, responsive UI

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your model file**
   `app.py` models ko do possible locations me dhoondta hai:
   - Preferred: `c:/cattle_app/breed_model.pth` and `c:/cattle_app/disease_model.pth`
   - Alternative: `c:/breed_model.pth` and `c:/disease_model.pth` (parent directory of app.py)

   Agar dono me se koi bhi missing hai, server start ho jayega but prediction endpoints error JSON denge.

3. **Run the server**
   ```bash
   cd cattle_app
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:5000`

## API Endpoints

### `POST /predict`
Upload an image for breed classification.
- **Body**: `multipart/form-data` with field `image`
- **Returns**: JSON with top-5 predictions and confidence scores

```json
{
  "success": true,
  "top_breed": "Gir",
  "top_confidence": 94.3,
  "predictions": [
    {"breed": "gir", "breed_display": "Gir", "confidence": 94.3},
    ...
  ]
}
```

### `GET /breeds`
Returns all 73 supported breed names.

## Supported Breeds
Alambadi, Amritmahal, Ayrshire, Bachaur, Badri, Banni, Bargur, Belahi, Bhadawari, Brown Swiss, Chhattisgarhi, Chilika, Dagri, Dangi, Deoni, Gangatiri, Gaolao, Ghumusari, Gir, Gojri, Guernsey, Hallikar, Hariana, Himachali Pahari, Holstein-Friesian, Jaffarabadi, Jersey, Kalahandi, Kangayam, Kankrej, Kenkatha, Khariar, Kherigarh, Khillar, Konkan Kapila, Kosali, Krishna Valley, Ladakhi, Lakhimi, Limousin, Luit (Swamp), Malnad Gidda, Malvi, Marathwadi, Mehsana, Mewati, Motu, Murrah, Nagori, Nagpuri, Nari, Nili Ravi, Nimari, Ongole, Pandharpuri, Poda Thurpu, Ponwar, Pulikulam, Punganur, Purnea, Rathi, Red Dane, Red Kandhari, Red Sindhi, Sahiwal, Shweta Kapila, Siri, Surti, Tharparkar, Thutho, Toda, Umblachery, Vechur
