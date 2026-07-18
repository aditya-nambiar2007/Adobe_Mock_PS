import { useState, useMemo, useEffect, useCallback } from 'react'
import {
  ReactFlow,
  Background,
  BaseEdge,
  getBezierPath,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import dagre from 'dagre'
import BaseNode from './BaseNode'
import { history } from '../utilities/indexedDB'

function flattenTree(node, edges, currId) {
  const nodes = []
  const children = node.children || []
  const currentJobId = node.jobId || node.job_id
  nodes.push({
    id: node.id,
    type: 'historyNode',
    data: { title: node.label, subtitle: node.time, id: node.id, currId, jobId: currentJobId },
  })
  for (const child of children) {
    const childJobId = child.jobId || child.job_id
    const hasSameJob = currentJobId && childJobId && currentJobId === childJobId
    const edgeObj = {
      id: `${node.id}->${child.id}`,
      source: node.id,
      target: child.id,
      type: 'historyEdge',
    }
    if (hasSameJob) {
      edgeObj.style = { stroke: '#ff7f50', strokeWidth: 3, strokeDasharray: 'none' }
      edgeObj.markerEnd = {
        type: MarkerType.ArrowClosed,
        color: '#ff7f50',
        width: 15,
        height: 15,
      }
    }
    edges.push(edgeObj)
    const [childNodes] = flattenTree(child, edges, currId)
    nodes.push(...childNodes)
  }
  return [nodes, edges]
}

const nodeWidth = 180
const nodeHeight = 52

function layoutNodes(rawNodes, rawEdges) {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', ranksep: 200, nodesep: 80, marginx: 40, marginy: 40 })
  for (const node of rawNodes) {
    g.setNode(node.id, { width: nodeWidth, height: nodeHeight })
  }
  for (const edge of rawEdges) {
    g.setEdge(edge.source, edge.target)
  }
  dagre.layout(g)
  return rawNodes.map((node) => {
    const pos = g.node(node.id)
    return { ...node, position: { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 } }
  })
}

function HistoryNode({ data, selected }) {
  return (
    <BaseNode data={data} selected={selected} currId={data.currId}>
      <div className="history-node__subtitle">{data.subtitle}</div>
    </BaseNode>
  )
}

function HistoryEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  style,
  markerEnd,
}) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      className={`flow-edge${selected ? ' flow-edge--selected' : ''}`}
      style={style}
      markerEnd={markerEnd}
    />
  )
}

const nodeTypes = { historyNode: HistoryNode }
const edgeTypes = { historyEdge: HistoryEdge }
const defaultEdgeOptions = {
  type: 'historyEdge',
}


const connectionLineStyle = { stroke: '#FF1E8A', strokeWidth: 2, strokeDasharray: '5 5' }

function FlowCanvas({ miniature, treeData, currNode, setNode }) {
  const { initialNodes, initialEdges } = useMemo(() => {
    const edges = []
    const [rawNodes] = flattenTree(treeData, edges, currNode?.id)
    return { initialNodes: layoutNodes(rawNodes, edges), initialEdges: edges }
  }, [treeData, currNode?.id])
  
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  
  useEffect(() => {
    setNodes((nds) => {
      const currentIds = new Set(nds.map((n) => n.id))
      const match =
        initialNodes.length === nds.length &&
        initialNodes.every((n) => currentIds.has(n.id))

      if (!match) {
        return initialNodes.map((node) => ({
          ...node,
          selected: node.id === currNode?.id,
        }))
      }

      return nds.map((node) => ({
        ...node,
        data: { ...node.data, currId: currNode?.id },
        selected: node.id === currNode?.id,
      }))
    })
  }, [initialNodes, currNode?.id, setNodes])

  useEffect(() => {
    setEdges(initialEdges)
  }, [initialEdges, setEdges])

  const handleNodeClick = useCallback(
    (event, node) => {
      if (setNode) {
        setNode(node.id)
      }
    },
    [setNode]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      defaultEdgeOptions={defaultEdgeOptions}
      connectionLineStyle={connectionLineStyle}
      onNodeClick={handleNodeClick}
      fitView
      fitViewOptions={{
        padding: miniature ? 0.6 : 0.25,
      }}
      minZoom={miniature ? 0.05 : 0.1}
      maxZoom={miniature ? 0.5 : 3}
      nodesDraggable={!miniature}
      nodesConnectable={false}
      panOnDrag={!miniature}
      zoomOnScroll={!miniature}
      panOnScroll={false}
      colorMode="dark"
      proOptions={{ hideAttribution: true }}
    >
      {!miniature && <Background variant="dots" gap={20} size={1.5} color="#333" />}
    </ReactFlow>
  )
}

export default function TreePanel({headId, historyVersion, setNode, currNode}) {
  const [fullscreen, setFullscreen] = useState(false)
  const [treeData,setTree]=useState(null)
  
  const close = useCallback(() => setFullscreen(false), [])
  
  useEffect(() => {
    if (!fullscreen) return
    const handler = (e) => { if (e.key === 'Escape') setFullscreen(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [fullscreen])
  
  useEffect(() => {
    let cancelled = false

    history.getTree(headId)
      .then((data) => {
        if (!cancelled) {
          setTree(data)
        }
      })
      .catch((err) => {
        console.error(err)
      })

    return () => {
      cancelled = true
    }
  }, [headId, historyVersion])
  
  return (
    <>
      {fullscreen && <div className="tree-panel-backdrop" onClick={close} />}
      <aside className={`tree-panel${fullscreen ? ' tree-panel--fullscreen' : ''}`}>
        <div className="tree-panel-header">
          <h2>Edit History</h2>
          <button
            className="fullscreen-btn"
            onClick={() => setFullscreen(!fullscreen)}
            title={fullscreen ? 'Minimize' : 'Full screen'}
          >
            {fullscreen ? (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M4 14L2 12M2 12L4 10M2 12H6M12 2L14 4M14 4L12 6M14 4H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 4L4 2M4 2L2 6M6 14L4 12M12 2L14 6M10 14L12 12M14 10L12 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.4"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M2 9L7 14L14 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </button>
        </div>
        <div className="tree-container">
          {treeData ? (
            <FlowCanvas key={fullscreen ? 'full' : 'mini'} miniature={!fullscreen} treeData={treeData} currNode={currNode} setNode={setNode}/>
          ) : (
            <div style={{ padding: '1rem', color: '#888', fontSize: '0.85rem' }}>
              No edit history yet. Upload an image to get started.
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
