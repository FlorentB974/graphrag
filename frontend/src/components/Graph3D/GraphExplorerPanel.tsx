'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { ArrowPathIcon, AdjustmentsHorizontalIcon, ExclamationCircleIcon, SparklesIcon } from '@heroicons/react/24/outline'

import { GraphScene } from './GraphScene'
import { api } from '@/lib/api'
import type { GraphVisualizationData } from '@/types'
import { ForceSimulation3D, GraphLayout, defaultForceConfig } from '@/lib/graph/forceSimulation'

interface GraphExplorerPanelProps {
  onRefreshRequest?: () => void
}

export function GraphExplorerPanel({ onRefreshRequest }: GraphExplorerPanelProps) {
  const [graphData, setGraphData] = useState<GraphVisualizationData | null>(null)
  const [layout, setLayout] = useState<GraphLayout | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set())
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null)
  const [minWeight, setMinWeight] = useState(0)

  const loadGraph = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getGraphVisualization()
      setGraphData(data)

      const simulation = new ForceSimulation3D(defaultForceConfig)
      const newLayout = await simulation.generateLayout(data)
      setLayout(newLayout)
    } catch (err) {
      console.error('Failed to load graph visualization data', err)
      setError(err instanceof Error ? err.message : 'Unable to load graph data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadGraph()
  }, [loadGraph])

  useEffect(() => {
    const handler = () => loadGraph()
    window.addEventListener('documents:processed', handler)
    window.addEventListener('documents:uploaded', handler)
    window.addEventListener('documents:processing-updated', handler)
    return () => {
      window.removeEventListener('documents:processed', handler)
      window.removeEventListener('documents:uploaded', handler)
      window.removeEventListener('documents:processing-updated', handler)
    }
  }, [loadGraph])

  const availableTypes = useMemo(() => {
    if (!graphData) return [] as string[]
    const unique = new Set<string>()
    graphData.entities.forEach((e) => unique.add(e.type || 'unknown'))
    return Array.from(unique).sort()
  }, [graphData])

  const availableLevels = useMemo(() => {
    if (!graphData) return [] as number[]
    const levels = new Set<number>()
    graphData.communities.forEach((community) => {
      if (community.level > 0) levels.add(community.level)
    })
    return Array.from(levels).sort((a, b) => a - b)
  }, [graphData])

  const toggleType = (type: string) => {
    setSelectedTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) {
        next.delete(type)
      } else {
        next.add(type)
      }
      return next
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-secondary-900 dark:text-secondary-50">Interactive 3D Knowledge Graph</h3>
          <p className="text-xs text-secondary-600 dark:text-secondary-400">Communities, filters, and smart sizing powered by your ingested entities.</p>
        </div>
        <div className="flex gap-2">
          <button
            className="button-secondary flex items-center gap-2"
            onClick={() => {
              loadGraph()
              onRefreshRequest?.()
            }}
            disabled={loading}
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
        <div className="card p-3 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-secondary-700 dark:text-secondary-300 text-sm font-medium">
            <SparklesIcon className="w-4 h-4" />
            Search & Highlight
          </div>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Find entities..."
            className="w-full rounded-md border border-secondary-300 bg-white px-3 py-2 text-sm shadow-inner focus:outline-none focus:ring-1 focus:ring-primary-500 dark:bg-secondary-800 dark:border-secondary-700"
          />
        </div>

        <div className="card p-3 flex flex-col gap-2 lg:col-span-2">
          <div className="flex items-center gap-2 text-secondary-700 dark:text-secondary-300 text-sm font-medium">
            <AdjustmentsHorizontalIcon className="w-4 h-4" />
            Filters
          </div>
          <div className="flex flex-wrap gap-2">
            {availableTypes.map((type) => {
              const active = selectedTypes.has(type)
              return (
                <button
                  key={type}
                  onClick={() => toggleType(type)}
                  className={`px-3 py-1 text-xs rounded-full border ${
                    active
                      ? 'border-primary-500 bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:border-primary-500'
                      : 'border-secondary-300 text-secondary-700 hover:border-primary-400'
                  }`}
                >
                  {type}
                </button>
              )
            })}
            {availableTypes.length === 0 && <p className="text-xs text-secondary-500">No entity types detected yet.</p>}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-secondary-600 dark:text-secondary-400">
            <label className="flex items-center gap-2">
              <span className="font-medium">Community level</span>
              <select
                value={selectedLevel ?? ''}
                onChange={(e) => setSelectedLevel(e.target.value ? Number(e.target.value) : null)}
                className="rounded-md border border-secondary-300 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-primary-500 dark:bg-secondary-800 dark:border-secondary-700"
              >
                <option value="">All</option>
                {availableLevels.map((level) => (
                  <option key={level} value={level}>
                    Level {level}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="font-medium">Min edge weight</span>
              <input
                type="range"
                min={0}
                max={5}
                step={0.1}
                value={minWeight}
                onChange={(e) => setMinWeight(Number(e.target.value))}
              />
              <span className="w-10 text-right">{minWeight.toFixed(1)}</span>
            </label>
          </div>
        </div>

        <div className="card p-3 text-xs text-secondary-600 dark:text-secondary-400 space-y-1">
          <p className="font-semibold text-secondary-900 dark:text-secondary-100">Graph Stats</p>
          <p>Entities: {graphData?.entities.length ?? '—'}</p>
          <p>Relationships: {graphData?.relationships.length ?? '—'}</p>
          <p>Communities: {graphData ? Math.max(0, graphData.communities.length - 1) : '—'}</p>
        </div>
      </div>

      <div className="card p-4">
        {error && (
          <div className="flex items-center gap-2 text-red-600 text-sm mb-3">
            <ExclamationCircleIcon className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-20 text-secondary-600 dark:text-secondary-300">
            <ArrowPathIcon className="w-5 h-5 animate-spin mr-2" />
            Building the galaxy view...
          </div>
        )}

        {!loading && layout && graphData && (
          <GraphScene
            layout={layout}
            searchTerm={searchTerm}
            selectedTypes={selectedTypes}
            selectedLevel={selectedLevel}
            minWeight={minWeight}
          />
        )}

        {!loading && !layout && !error && (
          <div className="text-center py-12 text-secondary-600 dark:text-secondary-400">
            <p className="text-sm">No graph data available yet.</p>
            <p className="text-xs">Process documents with entity extraction to unlock the 3D explorer.</p>
          </div>
        )}
      </div>
    </div>
  )
}
