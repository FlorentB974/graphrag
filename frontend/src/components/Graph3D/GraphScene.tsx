'use client'

import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls, Sphere } from '@react-three/drei'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

import GalaxyBackground from './GalaxyBackground'
import type { GraphLayout, Link3D, Node3D } from '@/lib/graph/forceSimulation'
import { calculateLinkThickness } from '@/lib/graph/forceSimulation'

interface GraphSceneProps {
  layout: GraphLayout
  searchTerm: string
  selectedTypes: Set<string>
  selectedLevel: number | null
  minWeight: number
}

function GraphNode({
  node,
  highlighted,
  faded,
  onHover,
}: {
  node: Node3D
  highlighted: boolean
  faded: boolean
  onHover: (node: Node3D | null) => void
}) {
  const ref = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (ref.current) {
      const targetScale = highlighted ? 1.3 : 1
      ref.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1)
    }
  })

  return (
    <group position={[node.x, node.y, node.z]}>
      <Sphere
        ref={ref}
        args={[Math.max(0.6, node.computedSize), 28, 28]}
        onPointerOver={() => onHover(node)}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          color={node.computedColor}
          emissive={new THREE.Color(node.computedColor)}
          emissiveIntensity={highlighted ? 1.2 : 0.15}
          transparent
          opacity={faded ? 0.35 : 0.95}
        />
      </Sphere>
    </group>
  )
}

function GraphLink({ link, emphasize }: { link: Link3D; emphasize: boolean }) {
  const thickness = calculateLinkThickness(link.weight) * (emphasize ? 1.6 : 1)
  const color = emphasize ? '#ffffff' : '#a0aec0'
  const opacity = emphasize ? 0.95 : 0.65

  return (
    <Line
      points={[
        [link.source.x, link.source.y, link.source.z],
        [link.target.x, link.target.y, link.target.z],
      ]}
      color={color}
      lineWidth={thickness}
      transparent
      opacity={opacity}
    />
  )
}

export function GraphScene({ layout, searchTerm, selectedTypes, selectedLevel, minWeight }: GraphSceneProps) {
  const [hoveredNode, setHoveredNode] = useState<Node3D | null>(null)

  const searchMatches = useMemo(() => {
    if (!searchTerm.trim()) return new Set<string>()
    const term = searchTerm.toLowerCase()
    const matches = new Set<string>()
    layout.nodes.forEach((node) => {
      if (node.title.toLowerCase().includes(term) || node.description.toLowerCase().includes(term)) {
        matches.add(node.id)
      }
    })
    return matches
  }, [layout.nodes, searchTerm])

  const filteredNodes = useMemo(() => {
    return layout.nodes.filter((node) => {
      if (selectedTypes.size > 0 && !selectedTypes.has(node.type)) return false
      if (selectedLevel !== null && node.communityLevel !== selectedLevel) return false
      return true
    })
  }, [layout.nodes, selectedLevel, selectedTypes])

  const visibleNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes])

  const filteredLinks = useMemo(() => {
    return layout.links.filter((link) => {
      if (link.weight < minWeight) return false
      return visibleNodeIds.has(link.source.id) && visibleNodeIds.has(link.target.id)
    })
  }, [layout.links, minWeight, visibleNodeIds])

  const visibleCommunities = useMemo(() => {
    const ids = new Set(filteredNodes.map((node) => node.community?.id).filter(Boolean) as string[])
    return layout.communities.filter((community) => community.level > 0 && ids.has(community.id))
  }, [filteredNodes, layout.communities])

  const cameraPosition: [number, number, number] = [0, 0, 420]

  return (
    <div className="relative">
      <Canvas camera={{ position: cameraPosition, fov: 55 }} className="rounded-lg bg-secondary-950/60">
        <color attach="background" args={["#05060b"]} />
        <GalaxyBackground />

        <ambientLight intensity={0.35} />
        <pointLight position={[100, 120, 140]} intensity={1.2} color="#9ad5ff" />
        <pointLight position={[-120, -140, -160]} intensity={0.6} color="#ff9bd5" />

        {visibleCommunities.map((community) => {
          const bounds = community.computedBounds
          if (!bounds) return null
          return (
            <mesh key={community.id} position={bounds.center}>
              <boxGeometry args={[bounds.size[0], bounds.size[1], bounds.size[2]]} />
              <meshBasicMaterial
                color={community.computedColor || '#4ecdc4'}
                transparent
                opacity={community.computedOpacity ?? 0.12}
                wireframe
              />
            </mesh>
          )
        })}

        {filteredLinks.map((link) => {
          const emphasize = searchMatches.has(link.source.id) || searchMatches.has(link.target.id)
          return <GraphLink key={link.id} link={link} emphasize={emphasize} />
        })}

        {filteredNodes.map((node) => (
          <GraphNode
            key={node.id}
            node={node}
            highlighted={hoveredNode?.id === node.id || searchMatches.has(node.id)}
            faded={searchMatches.size > 0 && !searchMatches.has(node.id)}
            onHover={setHoveredNode}
          />
        ))}

        <OrbitControls enableDamping dampingFactor={0.08} minDistance={60} maxDistance={600} autoRotate autoRotateSpeed={0.5} />
      </Canvas>

      {hoveredNode && (
        <div className="absolute left-4 bottom-4 max-w-xs rounded-lg bg-secondary-900/80 text-secondary-50 shadow-xl backdrop-blur p-4 border border-secondary-700">
          <p className="text-sm font-semibold mb-1">{hoveredNode.title}</p>
          <p className="text-xs text-secondary-200 capitalize">{hoveredNode.type}</p>
          <p className="text-xs mt-2 line-clamp-4 text-secondary-100">{hoveredNode.description || 'No description available.'}</p>
          <div className="text-[11px] text-secondary-300 mt-2 flex gap-3">
            <span>Degree: {hoveredNode.degree}</span>
            <span>Frequency: {hoveredNode.frequency}</span>
          </div>
        </div>
      )}
    </div>
  )
}
