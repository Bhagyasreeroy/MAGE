# Kubernetes Manifests — Placeholder

Kubernetes deployment manifests for MAGE will be added in a future milestone.

## Planned Manifests

```
infra/k8s/
├── namespace.yaml             # mage namespace
├── backend/
│   ├── deployment.yaml        # FastAPI backend Deployment
│   ├── service.yaml           # ClusterIP Service
│   └── hpa.yaml               # HorizontalPodAutoscaler
├── frontend/
│   ├── deployment.yaml        # Next.js frontend Deployment
│   └── service.yaml           # LoadBalancer Service
├── chromadb/
│   ├── statefulset.yaml       # ChromaDB StatefulSet
│   ├── pvc.yaml               # PersistentVolumeClaim
│   └── service.yaml           # ClusterIP Service
├── ingress/
│   └── ingress.yaml           # NGINX Ingress (TLS termination)
└── secrets/
    └── secrets.yaml.example   # Secret template (never commit real secrets)
```

## Prerequisites

- Kubernetes cluster (local: `minikube` or `kind`; cloud: GKE / EKS / AKS)
- `kubectl` configured
- Helm 3 (for ingress-nginx, cert-manager)

## Deployment Steps (future)

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/chromadb/
kubectl apply -f infra/k8s/backend/
kubectl apply -f infra/k8s/frontend/
kubectl apply -f infra/k8s/ingress/
```
