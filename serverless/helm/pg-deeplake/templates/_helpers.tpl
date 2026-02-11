{{/*
Expand the name of the chart.
*/}}
{{- define "pg-deeplake.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "pg-deeplake.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "pg-deeplake.labels" -}}
helm.sh/chart: {{ include "pg-deeplake.name" . }}
app.kubernetes.io/name: {{ include "pg-deeplake.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "pg-deeplake.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pg-deeplake.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Validate secure defaults for production-like deployments.
*/}}
{{- define "pg-deeplake.validate" -}}
{{- if .Values.security.enforceSecureDefaults -}}
  {{- if and .Values.security.requireEnvFromSecret (not .Values.envFromSecret) -}}
    {{- fail "security.requireEnvFromSecret=true requires envFromSecret to be set" -}}
  {{- end -}}
  {{- if and (not .Values.envFromSecret) (or (eq (trim .Values.postgres.password) "") (eq .Values.postgres.password "postgres")) -}}
    {{- fail "security.enforceSecureDefaults=true blocks empty/default postgres.password when envFromSecret is not set" -}}
  {{- end -}}
{{- end -}}
{{- end }}
