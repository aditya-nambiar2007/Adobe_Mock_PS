# Adobe Mock PS — Frontend

React + Vite frontend for the AI image editor. Built with Fabric.js canvas and React Flow.

## Quick Start

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`. Requires the backend running on `http://localhost:8000`.

### Production build

```bash
npm run build    # outputs to dist/
npm run preview  # preview the production build
```

---

## Project Structure

```
src/
├── main.jsx                 # React entry point
├── App.jsx                  # Root component: layout, state, routing
├── App.css                  # All component styles
├── index.css                # CSS variables, dark theme, reset
│
├── components/
│   ├── CanvasEditor.jsx     # Fabric.js canvas (drawing, filters, crop, resize)
│   ├── ChatWindow.jsx       # Chat sidebar: prompt input, results, explanations
│   ├── ChatHistoryView.jsx  # List of chat sessions
│   ├── LibraryView.jsx      # All saved images grid
│   ├── ToolbarRibbon.jsx    # Left toolbar (tools + actions)
│   ├── ToolOptions.jsx      # Right sidebar (tool settings)
│   ├── TreePanel.jsx        # ReactFlow edit history tree
│   ├── BaseNode.jsx         # Generic ReactFlow node wrapper
│   ├── NodeHandle.jsx       # Port handles for flow nodes
│   ├── ImageViewer.jsx      # Simple fallback image display
│   ├── StatusBar.jsx        # Bottom bar (zoom, dims, cursor)
│   └── ColorPicker.jsx      # Color swatch panel
│
└── utilities/
    ├── indexedDB.js         # IndexedDB stores: Images, History, Messages
    └── type.js              # dataURL / Blob conversion helpers
```

---

## Component Overview

### App.jsx (Root)

Central state management for the entire application. Key state:

| State | Purpose |
|---|---|
| `imageUrl` | Current image displayed on canvas (data URL) |
| `imageHistoryNode` | Current history node (imageId, label, prev/next links) |
| `head` | Root node of the current chat session |
| `view` | Active view: `editor` / `library` / `chatHistory` |
| `activeTool` | Current canvas tool: `select`, `brush`, `eraser`, `crop`, etc. |

Key callbacks passed to children:

- `handleUpload({url, filename})` — Loads an uploaded image
- `handleEditComplete(dataUri, label, filename)` — Stores edit result in IndexedDB + creates history node
- `setNode(nodeId)` — Navigate to a specific history node

### CanvasEditor.jsx

Wraps Fabric.js canvas. Key features:

| Feature | Implementation |
|---|---|
| Drawing | Brush, Eraser, Blur brush, Restore brush, Doodle eraser |
| Shapes | Rect, Circle, Line, Text |
| Image ops | Crop (rect selection), Resize (numbered inputs) |
| Filters | Brightness, Contrast, Saturation, Hue, Blur (Fabric.js filters) |
| Navigation | Pan (space+drag), Zoom (scroll) |
| History | Undo/redo stack (50 states), per-node preservation via `nodeHistoryMapRef` |
| Export | `canvasRef.current.exportImage()` → PNG data URL |

**Tools** — Set via `activeTool` prop. Keyboard shortcuts: `V`=select, `B`=brush, `E`=eraser, `T`=text, `U`=shape, `C`=crop, `L`=blur, `R`=restore, `D`=doodle-eraser. Ctrl+Z/Y=undo/redo, Ctrl+S=save.

### ChatWindow.jsx

Chat interface for sending edit prompts to the backend. Flow:

1. User types prompt, clicks Send
2. Saves any pending canvas changes first
3. `fetch('/edit', { body: { prompt, image } })`
4. On success: displays "Applied: prompt" message with explanation section

**Explanation display** — Each assistant message with an edit result shows:

```
Applied: remove the person

What changed: The person was identified and removed from the image.
             The background was preserved.

Technical Details:
- Person segmented using SAM.
- Object removed via inpainting.

Show Execution Details ▼
  segment → success → 9405ms
    Target: person
    Model: sam-vit-base
  remove  → success → 14423ms
    Model: sd-inpaint
```

The execution log is collapsible to keep the UI clean.

### TreePanel.jsx

ReactFlow-based history tree showing all edit nodes. Layout via dagre. Supports:

- Click to navigate to any node
- Fullscreen modal view
- Visual indicators for current node (accent border)

### Data Persistence

**IndexedDB** (via `indexedDB.js`) — 3 object stores:

| Store | Key | Value |
|---|---|---|
| `Images` | UUID | Image blob |
| `History` | Node ID | `{label, time, imageId, filename, prevNode, nextNode[]}` |
| `Messages` | Head ID | `{messages[]}` |

**localStorage** — Chat session metadata keyed by `adobe_mock_ps_chats`.

---

## Backend Integration

The frontend talks to the backend via REST at `/edit`, `/upload`, `/result/{id}`, `/history/{id}`.

### Edit flow

```javascript
// ChatWindow.jsx handleSend()
const res = await fetch('/edit', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt, image: canvasURI }),
});
const data = await res.json();

// data contains:
//   final_image: base64 PNG
//   steps: [{operation, image, duration_ms, details}]
//   execution_log: [{operation, target, model, status, duration, ...}]
//   explanation: {plain_english, technical_summary, changes}
```

The `dataUrl` from `final_image` is stored in IndexedDB and shown on the canvas.

### Error handling

Network errors show "Edit failed. Is the backend running?" Backend errors display the error detail from the response.

---

## CSS Architecture

| File | Purpose |
|---|---|
| `index.css` | CSS variables (`--bg`, `--text`, `--accent`, `--node-bg`, etc.), dark theme defaults, reset |
| `App.css` | All component-specific styles (~1380 lines) |

Themes use CSS custom properties. The dark theme in `index.css` uses:

```css
--bg: #0f0f0f;
--accent: #FF1E8A;
--text: #e0e0e0;
--node-bg: #1a1a1a;
--border: #2a2a2a;
```

---

## Extending the Frontend

### Add a new tool

1. Add the tool ID to `TOOLS_WITH_OPTIONS` in `ToolOptions.jsx`
2. Add keyboard shortcut in `App.jsx` keydown handler
3. Implement the tool handler in `CanvasEditor.jsx` (see brush/crop patterns)
4. Add a toolbar button in `ToolbarRibbon.jsx`

### Add a new display section to edit results

Edit `ExplanationSection` in `ChatWindow.jsx`. The component receives `explanation` and `executionLog` props from API response.

### Add a new persistent store

Extend `indexedDB.js` with a new object store in the `dbPromise` upgrade handler.

---

## Tech Stack

| Library | Purpose |
|---|---|
| React 19 | UI framework |
| Vite 8 | Build tool and dev server |
| Fabric.js 7 | Canvas manipulation |
| @xyflow/react 12 | History tree visualization |
| dagre | Graph layout for tree |
| lucide-react | Icons |
| Oxlint | Rust-based linter |
