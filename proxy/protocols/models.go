package protocols

import (
	"encoding/json"
	"net/http"
	"sort"

	"github.com/lsongdev/miya-agents/openai"
)

// Models serves the OpenAI /v1/models API endpoint, listing the models of
// every registered provider.
func (env *Env) Models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var models []openai.Model
	var catalog []map[string]any
	seen := make(map[string]bool)
	for _, p := range env.ListProviders() {
		catalog = append(catalog, p.ModelCatalog...)
		ids := append([]string(nil), p.Models...)
		for id := range p.ModelAliases {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			if id == "" || seen[id] {
				continue
			}
			seen[id] = true
			models = append(models, openai.Model{
				ID:      id,
				Object:  "model",
				OwnedBy: p.Name,
			})
		}
	}
	resp := map[string]any{
		"object": "list",
		"data":   models,
	}
	if len(catalog) > 0 {
		// Codex CLI discovers richer model capabilities from this sibling field;
		// OpenAI-compatible clients continue to consume the standard data array.
		resp["models"] = catalog
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
