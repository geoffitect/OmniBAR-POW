# 🧬 OmniBAR Prompt Optimization Workbench

A complete real-time web interface for optimizing prompts using genetic-inspired mutation analysis and algorithmic validation.

## 🚀 Quick Start

### 1. One-Click Launch
```bash
python start_workbench.py
```

This will:
- ✅ Check all dependencies
- 🚀 Start the API server
- 🌐 Open the workbench in your browser
- 📊 Connect to the database

### 2. Start Optimizing Prompts
1. **Enter your prompt** in the text area
2. **Select exploration depth** (1-3, higher = more thorough)
3. **Click "Start Optimization"**
4. **Watch results stream in real-time!**

## 🎯 Key Features

### 🔄 Real-Time Optimization
- **Live progress tracking** - Watch variations generate in real-time
- **Streaming results** - See best scores update as they're found
- **Process control** - Start, stop, and monitor runs
- **Auto-completion** - Results load automatically when runs finish

### 📊 Advanced Visualization
- **Score-segregated panels** - Results grouped by validation performance
- **Mutation notation** - DNA-style compact change tracking (`source_6_root`)
- **Interactive histograms** - Protein and organism extraction frequencies
- **Hover tooltips** - Instant access to prompt mutations and scores

### 🗄️ Database Management
- **Browse all runs** - Navigate through optimization history
- **Auto-refresh** - Live updates when new data arrives
- **Run comparison** - Switch between different optimization attempts
- **Persistent storage** - All results saved to SQLite database

### 🧬 Mutation Analysis
Instead of showing full prompts, the workbench uses **compact mutation notation**:

- `source_6_root` = Word 6 changed from "source" to "root"
- `Extract_1_Distill` = Word 1 changed from "Extract" to "Distill"
- `word_3_DEL` = Word 3 was deleted
- `INS_5_new` = New word "new" inserted at position 5
- `ORIGINAL` = Identical to original prompt

This makes it **instantly clear** which specific changes led to better performance!

## 🏗️ Architecture

### Frontend (`frontend.html`)
- **React-style UI** with real-time updates
- **Plotly.js** for interactive visualizations
- **WebSocket-ready** for live streaming
- **Responsive design** for all screen sizes

### Backend (`api_server.py`)
- **Flask REST API** for all operations
- **Process management** for optimization runs
- **Real-time monitoring** with progress tracking
- **Database integration** with existing SQLite schema

### Integration Layer
- **Seamless connection** to existing `prompt_refiner_pydantic.py`
- **Visualization engine** from `visualize_prompt_landscape.py`
- **Backwards compatible** with existing database schemas

## 📡 API Endpoints

### Core Operations
- `GET /api/runs` - List all optimization runs
- `GET /api/runs/{timestamp}` - Get detailed run data
- `POST /api/start-run` - Start new optimization
- `POST /api/stop-run/{timestamp}` - Stop running process
- `GET /api/status/{timestamp}` - Real-time progress status

### Utilities
- `GET /api/health` - System health check
- `GET /api/generate-viz/{timestamp}` - Generate static visualization
- `GET /viz/{filename}` - Serve visualization files

## 🛠️ Installation & Setup

### Dependencies
```bash
pip install flask flask-cors plotly pydantic sqlite3
```

### Required Files
All these files must be in the same directory:
- `frontend.html` - Main web interface
- `api_server.py` - Backend API server
- `start_workbench.py` - Launcher script
- `prompt_refiner_pydantic.py` - Core optimization engine
- `visualize_prompt_landscape.py` - Visualization components

### Check Installation
```bash
python start_workbench.py --check
```

## 🎮 Usage Examples

### Basic Optimization
```bash
# Start the workbench
python start_workbench.py

# In the web interface:
# 1. Enter: "Extract proteins from this research paper"
# 2. Set depth: 2
# 3. Click "Start Optimization"
# 4. Watch real-time results!
```

### Advanced Configuration
```bash
# Custom port
python start_workbench.py --port 9000

# Debug mode
python start_workbench.py --debug

# Headless mode (no browser)
python start_workbench.py --no-browser
```

### Direct API Usage
```bash
# Start a run via API
curl -X POST http://localhost:8080/api/start-run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Extract proteins...", "depth": 2}'

# Check status
curl http://localhost:8080/api/status/20251003_134916

# List all runs
curl http://localhost:8080/api/runs
```

## 📈 Interpreting Results

### Score Panels
- **🟢 Perfect (100%):** All validation objectives achieved
- **🔵 Excellent (80-99%):** Near-perfect validation
- **🟡 Good (60-79%):** Decent performance
- **🔴 Poor (<60%):** Low validation scores

### Histogram Analysis
- **Bar length** = Frequency of extraction across variations
- **Hover tooltips** = Mutation notation + scores
- **Color coding** = Score range (perfect/excellent/good/poor)

### Mutation Patterns
Look for patterns in successful mutations:
- Which word positions are most important?
- What types of changes improve scores?
- Are there common successful transformations?

## 🔧 Troubleshooting

### Common Issues

**❌ Port already in use**
```bash
python start_workbench.py --port 9000
```

**❌ Missing dependencies**
```bash
pip install flask flask-cors plotly pydantic
```

**❌ Database not found**
- Run some optimizations first to create the database
- Or copy an existing `prompt_refiner_results.db` file

**❌ API connection failed**
- Check if the API server is running
- Verify the port in browser matches the server port
- Check firewall/proxy settings

### Debug Mode
Enable debug mode for detailed logging:
```bash
python start_workbench.py --debug
```

### Health Check
Verify system status:
```bash
curl http://localhost:8080/api/health
```

## 🎯 Best Practices

### Optimization Strategy
1. **Start with depth 1** for quick exploration
2. **Use depth 2** for standard optimization
3. **Reserve depth 3** for thorough analysis (slower)
4. **Monitor mutation patterns** to understand what works
5. **Compare across runs** to identify consistent improvements

### Performance Tips
- **Close unused browser tabs** during optimization
- **Use auto-refresh sparingly** for large databases
- **Stop runs cleanly** rather than forcing termination
- **Archive old databases** to keep the current one fast

### Workflow Recommendations
1. **Baseline first** - Run original prompt to establish baseline
2. **Iterative improvement** - Use insights from one run to inform the next
3. **Pattern recognition** - Look for successful mutation types
4. **Documentation** - Note which strategies work for your domain

## 🚀 Production Deployment

### Security Considerations
- **Run behind reverse proxy** (nginx/Apache) for production
- **Enable HTTPS** for secure connections
- **Restrict API access** if needed
- **Backup databases** regularly

### Scaling Options
- **Multiple workers** with gunicorn/uwsgi
- **Load balancing** for high traffic
- **Database clustering** for large datasets
- **Cloud deployment** on AWS/GCP/Azure

### Monitoring
- **Health checks** at `/api/health`
- **Process monitoring** for optimization runs
- **Database size** tracking
- **Performance metrics** collection

## 🧪 Extension Points

### Custom Validation
Extend `prompt_refiner_pydantic.py` with new Pydantic models for different domains.

### Additional Visualizations
Add new chart types to `visualize_prompt_landscape.py` and integrate with the frontend.

### API Enhancements
Extend `api_server.py` with new endpoints for specialized operations.

### Frontend Customization
Modify `frontend.html` to add domain-specific UI components.

## 📚 Technical Details

### Real-Time Architecture
- **Polling-based updates** every 2 seconds during active runs
- **Progressive enhancement** - works without JavaScript
- **Graceful degradation** when API is unavailable
- **Error recovery** with automatic retry logic

### Database Schema
Compatible with existing OmniBAR database:
- `test_runs` table for run metadata
- `prompt_variations` table for individual results
- **Backwards compatible** with older schema versions

### Visualization Engine
- **Server-side generation** of segmented data
- **Client-side rendering** with Plotly.js
- **Interactive tooltips** with mutation notation
- **Responsive layouts** for all screen sizes

---

## 🎉 Ready to Optimize!

The OmniBAR Workbench transforms prompt optimization from a command-line task into an **interactive, real-time experience**.

Start exploring the genetic landscape of your prompts today! 🧬✨