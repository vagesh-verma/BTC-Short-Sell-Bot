import React, { useEffect, useState, useRef } from 'react';
import { logger } from '../services/loggerService';
import { Terminal as TerminalIcon, X, Trash2, ChevronDown, ChevronUp } from 'lucide-react';

interface TerminalProps {
  serverLogs?: { timestamp: number; type: string; message: string }[];
}

export const Terminal: React.FC<TerminalProps> = ({ serverLogs = [] }) => {
  const [logs, setLogs] = useState(logger.getLogs());
  const [isOpen, setIsOpen] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsubscribe = logger.subscribe((newLogs) => {
      setLogs([...newLogs]);
    });
    return unsubscribe;
  }, []);

  const allLogs = [...logs, ...serverLogs].sort((a, b) => a.timestamp - b.timestamp);

  useEffect(() => {
    if (scrollRef.current && !isMinimized) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [allLogs, isMinimized]);

  if (!isOpen) return null;

  return (
    <div className={`fixed bottom-4 right-4 z-50 flex flex-col bg-[#0D0D0E] border border-white/10 rounded-xl shadow-2xl transition-all duration-300 ${isMinimized ? 'w-64 h-12' : 'w-[450px] h-[350px]'}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-white/5 rounded-t-xl">
        <div className="flex items-center gap-2">
          <TerminalIcon className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-white/70">Debug Console</span>
        </div>
        <div className="flex items-center gap-1">
          <button 
            onClick={() => setIsMinimized(!isMinimized)}
            className="p-1 hover:bg-white/10 rounded transition-colors"
          >
            {isMinimized ? <ChevronUp className="w-4 h-4 text-white/40" /> : <ChevronDown className="w-4 h-4 text-white/40" />}
          </button>
          <button 
            onClick={() => logger.clear()}
            className="p-1 hover:bg-white/10 rounded transition-colors"
            title="Clear logs"
          >
            <Trash2 className="w-4 h-4 text-white/40" />
          </button>
          <button 
            onClick={() => setIsOpen(false)}
            className="p-1 hover:bg-white/10 rounded transition-colors"
          >
            <X className="w-4 h-4 text-white/40" />
          </button>
        </div>
      </div>

      {/* Logs Body */}
      {!isMinimized && (
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 font-mono text-[11px] space-y-1.5 scrollbar-thin scrollbar-thumb-white/10"
        >
          {allLogs.length === 0 ? (
            <div className="text-white/20 italic text-center py-10">No activity logged yet...</div>
          ) : (
            allLogs.map((log, i) => (
              <div key={i} className="flex gap-2 animate-in fade-in slide-in-from-left-1 duration-300">
                <span className="text-white/20 shrink-0">[{new Date(log.timestamp).toLocaleTimeString([], { hour12: false })}]</span>
                <span className={
                  log.type === 'error' ? 'text-red-400' :
                  log.type === 'success' ? 'text-emerald-400' :
                  log.type === 'warning' ? 'text-amber-400' :
                  'text-white/70'
                }>
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};
