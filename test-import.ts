import { tradingService } from "./server/tradingService";
import { GRUModel } from "./server/modelService";
import { deltaRequest } from "./server/deltaApi";

console.log("Imports successful");
console.log("Trading Service instance:", !!tradingService);
process.exit(0);
