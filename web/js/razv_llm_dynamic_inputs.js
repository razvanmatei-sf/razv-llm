import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "razv.llm.dynamic.inputs",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RazvLLMChat") {
            return;
        }

        // Store original methods
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        // Override onNodeCreated to set up initial state
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            // Initialize image input counter
            this._imageInputCount = 0;

            // Ensure we have the base image input
            this.updateImageInputs();

            return result;
        };

        // Override onConnectionsChange to handle dynamic input creation
        nodeType.prototype.onConnectionsChange = function(type, slot, isConnected, link_info, output_slot) {
            const result = onConnectionsChange?.apply(this, arguments);

            // Only handle input connections for image slots
            if (type === 1) { // 1 = input, 0 = output
                const input = this.inputs[slot];

                // Check if this is an image input
                if (input && input.type === "IMAGE") {
                    if (isConnected) {
                        // When an image is connected, ensure we have enough slots
                        this.updateImageInputs();
                    } else {
                        // When disconnected, clean up unused slots (but keep at least one)
                        this.cleanupImageInputs();
                    }
                }
            }

            return result;
        };

        // Method to update image inputs - ensures we always have one empty slot
        nodeType.prototype.updateImageInputs = function() {
            // Count connected image inputs
            let connectedImageInputs = 0;
            let totalImageInputs = 0;

            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.type === "IMAGE") {
                    totalImageInputs++;
                    if (input.link !== null) {
                        connectedImageInputs++;
                    }
                }
            }

            // If all image inputs are connected, add a new one
            if (connectedImageInputs === totalImageInputs && totalImageInputs > 0) {
                const newInputName = totalImageInputs === 1 ? "image_2" : `image_${totalImageInputs + 1}`;
                this.addInput(newInputName, "IMAGE");
            }

            // If we don't have any image inputs, add the first one
            if (totalImageInputs === 0) {
                this.addInput("image", "IMAGE");
            }
        };

        // Method to clean up unused image inputs (keep at least one)
        nodeType.prototype.cleanupImageInputs = function() {
            // Find image inputs that are not connected
            const imagesToRemove = [];
            let imageInputCount = 0;
            let connectedCount = 0;

            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.type === "IMAGE") {
                    imageInputCount++;
                    if (input.link !== null) {
                        connectedCount++;
                    } else {
                        imagesToRemove.push(i);
                    }
                }
            }

            // Remove disconnected image inputs, but keep at least one image input
            // and always keep at least one disconnected slot for future connections
            const shouldKeep = Math.max(1, connectedCount + 1); // At least 1 total, or connected + 1 empty
            const currentTotal = imageInputCount;
            const toRemove = Math.max(0, currentTotal - shouldKeep);

            // Remove from the end (highest indices first to avoid index shifting issues)
            imagesToRemove.reverse();
            for (let i = 0; i < toRemove && i < imagesToRemove.length; i++) {
                this.removeInput(imagesToRemove[i]);
            }
        };
    }
});