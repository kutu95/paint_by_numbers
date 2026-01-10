# Local Development Setup

## Quick Start

**Start the backend:**
```bash
./START_BACKEND.sh
```

Or manually:
```bash
cd backend
source venv/bin/activate
python main.py
```

**Start the frontend** (in a new terminal):
```bash
cd frontend
npm run dev
```

## Starting the Backend

### Option 1: Using the provided script (Recommended)

```bash
./START_BACKEND.sh
```

### Option 2: Manual start

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
# venv\Scripts\activate  # On Windows

# Install dependencies if needed (only first time)
pip install -r requirements.txt

# Start the backend server
python main.py
# OR
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`

**Note:** The backend must be running on `http://localhost:8000` for the frontend to work in local development.

## Starting the Frontend

```bash
cd frontend

# Install dependencies if needed
npm install

# Start the development server
npm run dev
```

The frontend will run on `http://localhost:3000` by default.

## Environment Variables

### Backend
If you need to set environment variables for the backend (like `OPENAI_API_KEY`), create a `.env` file in the `backend` directory:

```bash
cd backend
echo "OPENAI_API_KEY=your-key-here" > .env
```

Or export them before running:
```bash
export OPENAI_API_KEY=your-key-here
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
The frontend uses `NEXT_PUBLIC_API_BASE_URL` to connect to the backend. Default is `http://localhost:8000`.

To override, create a `.env.local` file in the `frontend` directory:

```bash
cd frontend
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local
```

Then restart the frontend dev server.

## Troubleshooting

### "Connection refused" error
- **Check backend is running**: Open `http://localhost:8000` in your browser or run `curl http://localhost:8000/docs`
- **Check port is correct**: Backend should be on port 8000
- **Check firewall**: Make sure port 8000 isn't blocked

### Backend won't start
- **Check Python version**: Should be Python 3.8+
- **Check dependencies**: Run `pip install -r requirements.txt` in the backend directory
- **Check virtual environment**: Make sure it's activated

### Frontend can't connect
- **Check API_BASE_URL**: Should be `http://localhost:8000` for local development
- **Check CORS**: Backend should allow `http://localhost:3000` in CORS_ORIGINS
- **Check backend logs**: Look for CORS or connection errors
