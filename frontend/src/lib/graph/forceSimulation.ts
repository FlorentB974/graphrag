import * as d3 from 'd3-force-3d'
import type { GraphCommunity, GraphEntity, GraphVisualizationData } from '@/types'

export interface Node3D extends GraphEntity {
  x: number
  y: number
  z: number
  vx?: number
  vy?: number
  vz?: number
  community?: CommunityWithComputed
  communityLevel: number
  // Pre-computed values for performance optimization
  computedSize: number
  computedColor: string
}

export interface Link3D {
  id: string
  source: Node3D
  target: Node3D
  weight: number
  description: string
}

export interface GraphLayout {
  nodes: Node3D[]
  links: Link3D[]
  communities: CommunityWithComputed[]
}

export interface ForceConfig {
  chargeStrength: number
  linkDistance: number
  linkStrength: number
  collisionRadius: number
  communityStrength: number
  centerStrength: number
  spread3D: number
  levelSpacing: number
  sphericalConstraint: number
}

export const defaultForceConfig: ForceConfig = {
  chargeStrength: -100,
  linkDistance: 30,
  linkStrength: 0.2,
  collisionRadius: 6,
  communityStrength: 0.2, // Increased for stronger community clustering
  centerStrength: 0.02, // Lower center pull to maintain spherical shape
  spread3D: 150,
  levelSpacing: 40,
  sphericalConstraint: 0.05, // Strength of spherical positioning force
}

export type CommunityWithComputed = GraphCommunity & {
  computedBounds?: {
    center: [number, number, number]
    size: [number, number, number]
    padding: number
  }
  computedHierarchy?: {
    parentCommunities: GraphCommunity[]
    childCommunities: GraphCommunity[]
  }
  computedColor?: string
  computedOpacity?: number
}

const ENTITY_COLORS: Record<string, string> = {
  ORGANIZATION: '#ff6b6b',
  EVENT: '#4ecdc4',
  PERSON: '#45b7d1',
  GEO: '#96ceb4',
  PROCESS: '#feca57',
  unnamed: '#95a5a6',
}

const ENTITY_SIZES = {
  MIN: 2,
  MAX: 12,
  SCALE_FACTOR: 0.5,
}

const RELATIONSHIP_THICKNESS = {
  MIN: 0.5,
  MAX: 4,
  SCALE_FACTOR: 0.2,
}

export class ForceSimulation3D {
  private simulation: d3.Simulation<Node3D, undefined>
  private nodes: Node3D[] = []
  private links: Link3D[] = []
  private communities: CommunityWithComputed[] = []
  private config: ForceConfig
  private communityCenters: Map<string, { x: number; y: number; z: number; radius: number }> = new Map()

  constructor(config: ForceConfig = defaultForceConfig) {
    this.config = { ...config }
    this.simulation = d3
      .forceSimulation<Node3D>()
      .force('link', d3.forceLink<Node3D, d3.SimulationLinkDatum<Node3D>>().id((d: unknown) => (d as { id: string }).id))
      .force('charge', d3.forceManyBody<Node3D>())
      .force('center', d3.forceCenter())
      .force('collision', d3.forceCollide())
      .alphaDecay(0.03)
      .alphaMin(0.001)
  }

  generateLayout(graphData: GraphVisualizationData): Promise<GraphLayout> {
    return new Promise((resolve) => {
      this.preprocessData(graphData)
      this.setupForces()

      let iterationCount = 0
      const maxIterations = 500

      this.simulation.on('tick', () => {
        iterationCount++
        if (iterationCount >= maxIterations || this.simulation.alpha() < 0.001) {
          this.simulation.stop()
          // Pre-compute community data for performance optimization
          this.precomputeCommunityData()
          resolve({
            nodes: this.nodes,
            links: this.links,
            communities: this.communities,
          })
        }
      })

      this.simulation.nodes(this.nodes)
      ;(this.simulation.force('link') as d3.ForceLink<Node3D, Link3D>).links(this.links)
      this.simulation.restart()
    })
  }

  private preprocessData(graphData: GraphVisualizationData): void {
    this.communities = graphData.communities as CommunityWithComputed[]

    // Create entity lookup for community assignment
    const entityToCommunity = new Map<string, CommunityWithComputed>()
    this.communities.forEach((community) => {
      community.entity_ids.forEach((entityId) => {
        entityToCommunity.set(entityId, community)
      })
    })

    // Convert entities to 3D nodes with spherical knowledge universe distribution
    this.nodes = graphData.entities.map((entity, index) => {
      const community = entityToCommunity.get(entity.id)
      const communityLevel = community ? community.level : 0

      // Calculate abstraction level - higher degree + frequency = more central/abstract
      const abstractionScore = entity.degree + entity.frequency * 0.5
      const maxAbstraction = Math.max(...graphData.entities.map((e) => e.degree + e.frequency * 0.5))
      const minAbstraction = Math.min(...graphData.entities.map((e) => e.degree + e.frequency * 0.5))

      // Normalize abstraction to 0-1 scale (1 = most abstract, 0 = least abstract)
      const normalizedAbstraction =
        maxAbstraction > minAbstraction ? (abstractionScore - minAbstraction) / (maxAbstraction - minAbstraction) : 0.5

      // Calculate radius based on abstraction level (inverted - high abstraction = center)
      const minRadius = this.config.spread3D * 0.1 // Core radius
      const maxRadius = this.config.spread3D // Outer shell radius
      const radius = minRadius + (1 - normalizedAbstraction) * (maxRadius - minRadius)

      // Add community level influence to create sub-spheres
      const communityOffset = communityLevel * this.config.levelSpacing * 0.3
      const finalRadius = radius + communityOffset

      // Use Fibonacci sphere for even distribution on each radius shell
      const goldenAngle = Math.PI * (3 - Math.sqrt(5)) // Golden angle in radians
      const phi = Math.acos(1 - 2 * (index / graphData.entities.length)) // Uniform distribution
      const theta = goldenAngle * index

      // Add slight randomization to prevent perfect grid
      const randomFactor = 0.9 + Math.random() * 0.2
      const adjustedRadius = finalRadius * randomFactor

      return {
        ...entity,
        x: adjustedRadius * Math.sin(phi) * Math.cos(theta),
        y: adjustedRadius * Math.sin(phi) * Math.sin(theta),
        z: adjustedRadius * Math.cos(phi),
        community,
        communityLevel,
        abstractionLevel: normalizedAbstraction,
        computedSize: calculateNodeSize(entity.degree, entity.frequency),
        computedColor: ENTITY_COLORS[entity.type] || ENTITY_COLORS.unnamed,
      } as Node3D & { abstractionLevel: number }
    })

    const nodeMap = new Map<string, Node3D>()
    this.nodes.forEach((node) => {
      nodeMap.set(node.id, node)
      nodeMap.set(node.title, node)
    })

    this.links = graphData.relationships
      .map((rel) => {
        const sourceNode = nodeMap.get(rel.source)
        const targetNode = nodeMap.get(rel.target)

        if (!sourceNode || !targetNode) {
          console.warn(`Relationship link missing node: ${rel.source} -> ${rel.target}`)
          return null
        }

        return {
          id: rel.id,
          source: sourceNode,
          target: targetNode,
          weight: rel.weight,
          description: rel.description,
        }
      })
      .filter((link): link is Link3D => link !== null)
  }

  private setupForces(): void {
    const nodeChargeStrength = (d: Node3D) => {
      return this.config.chargeStrength - d.degree * 5
    }

    const linkDistance = (d: Link3D) => {
      const weightFactor = 1 / (d.weight * 0.05 + 1)
      return this.config.linkDistance * weightFactor
    }

    const linkStrength = (d: Link3D) => {
      return Math.min(1, this.config.linkStrength + d.weight * 0.015)
    }

    this.simulation
      .force(
        'link',
        d3
          .forceLink<Node3D, Link3D>()
          .id((d: unknown) => (d as { id: string }).id)
          .distance(linkDistance)
          .strength(linkStrength)
      )
      .force('charge', d3.forceManyBody<Node3D>().strength((d) => nodeChargeStrength(d as Node3D)))
      // Typings from d3-force are 2D-only; d3-force-3d accepts a z parameter for centering.
      .force('center', (d3 as unknown as any).forceCenter(0, 0, 0))
      .force(
        'collision',
        d3.forceCollide<Node3D>().radius((d) => this.config.collisionRadius + calculateNodeSize(d.degree, d.frequency))
      )
      .force('community', this.communityForce())
      .force('spherical', this.sphericalConstraint())
      .alpha(1)
  }

  private communityForce() {
    return (alpha: number) => {
      this.computeCommunityCenters()

      for (const node of this.nodes) {
        const community = node.community
        if (!community) continue

        const center = this.communityCenters.get(community.id)
        if (!center) continue

        const dx = node.x - center.x
        const dy = node.y - center.y
        const dz = node.z - center.z

        node.vx = (node.vx || 0) - dx * this.config.communityStrength * alpha
        node.vy = (node.vy || 0) - dy * this.config.communityStrength * alpha
        node.vz = (node.vz || 0) - dz * this.config.communityStrength * alpha
      }
    }
  }

  private sphericalConstraint() {
    return (alpha: number) => {
      for (const node of this.nodes) {
        const radialDistance = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z)
        if (radialDistance === 0) continue

        const desiredRadius = this.config.spread3D
        const offset = radialDistance - desiredRadius

        const strength = this.config.sphericalConstraint * alpha
        node.vx = (node.vx || 0) - (node.x / radialDistance) * offset * strength
        node.vy = (node.vy || 0) - (node.y / radialDistance) * offset * strength
        node.vz = (node.vz || 0) - (node.z / radialDistance) * offset * strength
      }
    }
  }

  private computeCommunityCenters(): void {
    this.communityCenters.clear()

    const communityData = new Map<string, { sumX: number; sumY: number; sumZ: number; count: number; maxRadius: number }>()

    this.nodes.forEach((node) => {
      const community = node.community
      if (!community) return

      const data = communityData.get(community.id) || { sumX: 0, sumY: 0, sumZ: 0, count: 0, maxRadius: 0 }
      data.sumX += node.x
      data.sumY += node.y
      data.sumZ += node.z
      data.count++
      const radius = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z)
      data.maxRadius = Math.max(data.maxRadius, radius)
      communityData.set(community.id, data)
    })

    communityData.forEach((data, communityId) => {
      const center = { x: data.sumX / data.count, y: data.sumY / data.count, z: data.sumZ / data.count }
      const radius = data.maxRadius + this.config.levelSpacing
      this.communityCenters.set(communityId, { ...center, radius })
    })
  }

  private precomputeCommunityData(): void {
    const baseRadius = this.config.spread3D * 0.4

    this.communities.forEach((community) => {
      const nodes = this.nodes.filter((node) => node.community?.id === community.id)
      if (nodes.length === 0) return

      let center: [number, number, number] = [0, 0, 0]
      nodes.forEach((node) => {
        center[0] += node.x
        center[1] += node.y
        center[2] += node.z
      })
      center = [center[0] / nodes.length, center[1] / nodes.length, center[2] / nodes.length]

      let maxDistance = 0
      nodes.forEach((node) => {
        const dx = node.x - center[0]
        const dy = node.y - center[1]
        const dz = node.z - center[2]
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
        maxDistance = Math.max(maxDistance, distance)
      })

      const parentInfluence = Math.max(0, 3 - community.level) * 0.3
      community.computedBounds = {
        center,
        size: [baseRadius + maxDistance * 1.5, baseRadius + maxDistance * 1.5, baseRadius + maxDistance * 1.5],
        padding: 20 + parentInfluence * 10,
      }
    })

    this.communities.forEach((community) => {
      const parentCommunities: GraphCommunity[] = []
      const childCommunities: GraphCommunity[] = []

      if (community.parent !== null && community.parent !== undefined) {
        const parentKey = typeof community.parent === 'string' ? community.parent : String(community.parent)
        const parentCommunity = this.communities.find((c) => String(c.human_readable_id) === parentKey)
        if (parentCommunity) {
          parentCommunities.push(parentCommunity)
        }
      }

      this.communities.forEach((otherCommunity) => {
        if (otherCommunity.parent !== null && otherCommunity.parent !== undefined) {
          const parentKey = typeof otherCommunity.parent === 'string' ? otherCommunity.parent : String(otherCommunity.parent)
          if (parentKey === String(community.human_readable_id)) {
            childCommunities.push(otherCommunity)
          }
        }
      })

      if (community.children && Array.isArray(community.children)) {
        community.children.forEach((childHumanId: string) => {
          const childCommunity = this.communities.find((c) => String(c.human_readable_id) === childHumanId)
          if (childCommunity && !childCommunities.some((c) => c.id === childCommunity.id)) {
            childCommunities.push(childCommunity)
          }
        })
      }

      community.computedHierarchy = {
        parentCommunities: parentCommunities.sort((a, b) => a.level - b.level),
        childCommunities: childCommunities.sort((a, b) => a.level - b.level),
      }

      const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
      community.computedColor = colors[community.level % colors.length]
      community.computedOpacity = 0.15
    })
  }

  stop(): void {
    if (this.simulation) {
      this.simulation.stop()
    }
  }
}

export function calculateNodeSize(degree: number, frequency: number): number {
  const size = ENTITY_SIZES.MIN + (degree + frequency * 0.1) * ENTITY_SIZES.SCALE_FACTOR
  return Math.min(size, ENTITY_SIZES.MAX)
}

export function calculateLinkThickness(weight: number): number {
  const thickness = RELATIONSHIP_THICKNESS.MIN + weight * RELATIONSHIP_THICKNESS.SCALE_FACTOR
  return Math.min(thickness, RELATIONSHIP_THICKNESS.MAX)
}
