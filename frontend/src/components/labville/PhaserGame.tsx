import Phaser from "phaser";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

import type { WorldState } from "../../api/client";
import { LabScene } from "./LabScene";

interface PhaserGameProps {
  world: WorldState | null;
  selectedRoomId: string | null;
  selectedAgentId: string | null;
  selectedEquipmentId: string | null;
  onSelectRoom: (id: string) => void;
  onSelectAgent: (id: string) => void;
  onSelectEquipment: (id: string) => void;
  actionFeedback: {
    nonce: number;
    agentId: string;
    verb: string;
    targetId: string | null;
    phase: "executing" | "resolved" | "blocked" | "failed";
    outcome: string;
  } | null;
}

export interface PhaserGameHandle {
  captureScene: () => Promise<string>;
}

const PhaserGame = forwardRef<PhaserGameHandle, PhaserGameProps>(function PhaserGame({
  world,
  selectedRoomId,
  selectedAgentId,
  selectedEquipmentId,
  onSelectRoom,
  onSelectAgent,
  onSelectEquipment,
  actionFeedback,
}: PhaserGameProps, ref): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const gameRef = useRef<Phaser.Game | null>(null);
  const sceneRef = useRef<LabScene | null>(null);
  const sceneReady = useRef(false);
  const pendingWorld = useRef<WorldState | null>(null);

  useImperativeHandle(ref, () => ({
    captureScene: async () => {
      if (!sceneReady.current || !sceneRef.current) {
        throw new Error("The replay scene is not ready yet.");
      }
      return sceneRef.current.captureScene();
    },
  }), []);

  // Store latest callbacks in refs so Phaser closures see current values
  const onSelectRoomRef = useRef(onSelectRoom);
  onSelectRoomRef.current = onSelectRoom;
  const onSelectAgentRef = useRef(onSelectAgent);
  onSelectAgentRef.current = onSelectAgent;
  const onSelectEquipmentRef = useRef(onSelectEquipment);
  onSelectEquipmentRef.current = onSelectEquipment;

  // Store latest props for deferred application after scene ready
  const latestWorld = useRef(world);
  latestWorld.current = world;
  const latestRoom = useRef(selectedRoomId);
  latestRoom.current = selectedRoomId;
  const latestAgent = useRef(selectedAgentId);
  latestAgent.current = selectedAgentId;
  const latestEquipment = useRef(selectedEquipmentId);
  latestEquipment.current = selectedEquipmentId;

  // Create game on mount
  useEffect(() => {
    if (!containerRef.current || gameRef.current) return;

    const labScene = new LabScene();
    sceneRef.current = labScene;

    labScene.setClickCallback((type, id) => {
      if (type === "room") onSelectRoomRef.current(id);
      else if (type === "agent") onSelectAgentRef.current(id);
      else onSelectEquipmentRef.current(id);
    });

    // When create() finishes, flush any pending state
    labScene.onceReady(() => {
      sceneReady.current = true;
      const w = pendingWorld.current ?? latestWorld.current;
      if (w) {
        labScene.syncWorld(w);
        pendingWorld.current = null;
      }
      if (latestRoom.current) labScene.selectRoom(latestRoom.current);
      if (latestAgent.current) labScene.selectAgent(latestAgent.current);
      if (latestEquipment.current) labScene.selectEquipment(latestEquipment.current);
    });

    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: containerRef.current,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      scene: [labScene],
      pixelArt: true,
      roundPixels: true,
      antialias: false,
      backgroundColor: "#2a2a2e",
      scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
      input: {
        mouse: { preventDefaultWheel: false },
        touch: { capture: true },
        activePointers: 3,
      },
    });
    gameRef.current = game;

    return () => {
      game.destroy(true);
      gameRef.current = null;
      sceneRef.current = null;
      sceneReady.current = false;
    };
  }, []);

  // Sync world state
  useEffect(() => {
    if (!world) return;
    if (sceneReady.current && sceneRef.current) {
      sceneRef.current.syncWorld(world);
    } else {
      pendingWorld.current = world;
    }
  }, [world]);

  // Sync selected room
  useEffect(() => {
    if (sceneReady.current && sceneRef.current) {
      sceneRef.current.selectRoom(selectedRoomId);
    }
  }, [selectedRoomId]);

  // Sync selected agent
  useEffect(() => {
    if (sceneReady.current && sceneRef.current) {
      sceneRef.current.selectAgent(selectedAgentId);
    }
  }, [selectedAgentId]);

  useEffect(() => {
    if (sceneReady.current && sceneRef.current) {
      sceneRef.current.selectEquipment(selectedEquipmentId);
    }
  }, [selectedEquipmentId]);

  useEffect(() => {
    if (sceneReady.current && sceneRef.current && actionFeedback) {
      sceneRef.current.showActionFeedback(actionFeedback);
    }
  }, [actionFeedback]);

  return <div className="labville-container" ref={containerRef} />;
});

export default PhaserGame;
