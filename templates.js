/*
 * radcopilot/ui/templates.js
 *
 * Canonical front-end template catalog for the modular RadCopilot refactor.
 *
 * Goals:
 * - mirror the server-side template registry in radcopilot/report/generator.py
 * - provide a stable browser-friendly API for template lookup and rendering
 * - remain usable both in plain browser scripts and CommonJS-based tooling
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.RadCopilotTemplates = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const DEFAULT_TEMPLATE_ID = 'ct-chest';

  const TEMPLATE_DEFINITIONS = {
    'ct-chest': {
      id: 'ct-chest',
      label: 'CT Chest',
      modality: 'ct-chest',
      description: 'Chest CT template optimized for pulmonary, pleural, mediastinal, and cardiac findings.',
      sectionOrder: ['lungs', 'pleura', 'mediastinum', 'heart', 'vasculature', 'chest_wall'],
      sectionLabels: {
        lungs: 'Lungs',
        pleura: 'Pleura',
        mediastinum: 'Mediastinum / Hila',
        heart: 'Heart / Pericardium',
        vasculature: 'Vasculature',
        chest_wall: 'Chest Wall / Osseous Structures'
      },
      sectionDefaults: {
        lungs: 'No focal air-space consolidation or suspicious acute pulmonary abnormality.',
        pleura: 'No pleural effusion or pneumothorax.',
        mediastinum: 'No pathologically enlarged mediastinal lymph nodes are identified.',
        heart: 'Heart size is within normal limits.',
        vasculature: 'No acute vascular abnormality is evident on this exam.',
        chest_wall: 'No acute osseous abnormality is evident on this exam.'
      },
      guidelineHint: 'Use pulmonary nodule follow-up language only when the findings clearly describe a lung nodule.',
      allowNegativesInImpression: false,
      checklist: [
        'Pulmonary parenchyma reviewed',
        'Pleural space reviewed',
        'Mediastinum and hila reviewed',
        'Heart and pericardium reviewed',
        'Major thoracic vasculature reviewed',
        'Chest wall and osseous structures reviewed'
      ]
    },

    'ct-abdomen-pelvis': {
      id: 'ct-abdomen-pelvis',
      label: 'CT Abdomen / Pelvis',
      modality: 'ct-abdomen-pelvis',
      description: 'Abdomen and pelvis CT template for solid organs, bowel, peritoneum, pelvic organs, and vasculature.',
      sectionOrder: ['liver', 'gallbladder', 'spleen', 'pancreas', 'adrenals', 'kidneys', 'bowel', 'peritoneum', 'pelvis', 'vasculature'],
      sectionLabels: {
        liver: 'Liver',
        gallbladder: 'Gallbladder / Biliary',
        spleen: 'Spleen',
        pancreas: 'Pancreas',
        adrenals: 'Adrenal Glands',
        kidneys: 'Kidneys / Urinary Tract',
        bowel: 'Bowel',
        peritoneum: 'Peritoneum / Mesentery',
        pelvis: 'Pelvic Organs',
        vasculature: 'Vasculature'
      },
      sectionDefaults: {
        liver: 'No acute hepatic abnormality is identified.',
        gallbladder: 'No acute biliary abnormality is identified.',
        spleen: 'No acute splenic abnormality is identified.',
        pancreas: 'No acute pancreatic abnormality is identified.',
        adrenals: 'No suspicious adrenal abnormality is identified.',
        kidneys: 'No hydronephrosis or other acute urinary tract abnormality is identified.',
        bowel: 'No bowel obstruction or focal acute inflammatory bowel process is identified.',
        peritoneum: 'No ascites or free intraperitoneal air.',
        pelvis: 'No acute pelvic abnormality is identified on this exam.',
        vasculature: 'No acute vascular abnormality is evident on this exam.'
      },
      guidelineHint: 'Do not apply pulmonary nodule guidance to adrenal or adnexal findings.',
      allowNegativesInImpression: false,
      checklist: [
        'Hepatobiliary organs reviewed',
        'Spleen and pancreas reviewed',
        'Adrenal glands reviewed',
        'Kidneys and urinary tract reviewed',
        'Bowel reviewed',
        'Peritoneum and mesentery reviewed',
        'Pelvic organs reviewed',
        'Abdominopelvic vasculature reviewed'
      ]
    },

    'xr-chest': {
      id: 'xr-chest',
      label: 'Chest X-Ray',
      modality: 'xr-chest',
      description: 'Chest radiograph template for pulmonary, pleural, cardiomediastinal, and osseous review.',
      sectionOrder: ['lungs', 'pleura', 'heart', 'mediastinum', 'chest_wall'],
      sectionLabels: {
        lungs: 'Lungs',
        pleura: 'Pleura',
        heart: 'Cardiomediastinal Silhouette',
        mediastinum: 'Mediastinum',
        chest_wall: 'Osseous Structures'
      },
      sectionDefaults: {
        lungs: 'No focal air-space opacity is identified.',
        pleura: 'No pleural effusion or pneumothorax.',
        heart: 'Cardiomediastinal silhouette is within normal size limits.',
        mediastinum: 'No acute mediastinal abnormality is evident on this exam.',
        chest_wall: 'No acute osseous abnormality is identified on this exam.'
      },
      guidelineHint: '',
      allowNegativesInImpression: false,
      checklist: [
        'Lungs reviewed',
        'Pleural spaces reviewed',
        'Cardiomediastinal silhouette reviewed',
        'Mediastinum reviewed',
        'Osseous structures reviewed'
      ]
    },

    'mri-brain': {
      id: 'mri-brain',
      label: 'MRI Brain',
      modality: 'mri-brain',
      description: 'Brain MRI template for intracranial findings, sinus review, and vascular flow voids.',
      sectionOrder: ['brain', 'sinuses', 'vasculature', 'general'],
      sectionLabels: {
        brain: 'Brain',
        sinuses: 'Paranasal Sinuses / Mastoids',
        vasculature: 'Vascular Flow Voids',
        general: 'Other'
      },
      sectionDefaults: {
        brain: 'No acute intracranial abnormality is identified on this exam.',
        sinuses: 'No acute paranasal sinus or mastoid abnormality is evident on this exam.',
        vasculature: 'Major intracranial vascular flow voids are preserved.',
        general: 'No additional acute finding is identified.'
      },
      guidelineHint: '',
      allowNegativesInImpression: false,
      checklist: [
        'Brain parenchyma reviewed',
        'Extra-axial spaces reviewed',
        'Paranasal sinuses and mastoids reviewed',
        'Major vascular flow voids reviewed',
        'Additional incidental findings reviewed'
      ]
    },

    generic: {
      id: 'generic',
      label: 'Generic Report',
      modality: 'generic',
      description: 'Generic fallback template for cases without a dedicated modality template.',
      sectionOrder: ['general'],
      sectionLabels: {
        general: 'Findings'
      },
      sectionDefaults: {
        general: 'No acute abnormality is identified on this exam.'
      },
      guidelineHint: '',
      allowNegativesInImpression: false,
      checklist: ['Findings reviewed for clinically significant abnormality']
    }
  };

  const MODALITY_ALIASES = {
    'ct chest': 'ct-chest',
    'ctchest': 'ct-chest',
    'chest ct': 'ct-chest',
    'chest xray': 'xr-chest',
    'chest x-ray': 'xr-chest',
    'xray chest': 'xr-chest',
    'x-ray chest': 'xr-chest',
    'cxr': 'xr-chest',
    'ct ap': 'ct-abdomen-pelvis',
    'ct a/p': 'ct-abdomen-pelvis',
    'ct abdomen pelvis': 'ct-abdomen-pelvis',
    'ct abdomen/pelvis': 'ct-abdomen-pelvis',
    'mri head': 'mri-brain',
    'brain mri': 'mri-brain'
  };

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function normalizeTemplateId(templateId) {
    const raw = String(templateId || '').trim().toLowerCase();
    if (!raw) return DEFAULT_TEMPLATE_ID;
    if (TEMPLATE_DEFINITIONS[raw]) return raw;
    return MODALITY_ALIASES[raw] || raw;
  }

  function getTemplate(templateId) {
    const normalized = normalizeTemplateId(templateId);
    return clone(TEMPLATE_DEFINITIONS[normalized] || TEMPLATE_DEFINITIONS.generic);
  }

  function hasTemplate(templateId) {
    const normalized = normalizeTemplateId(templateId);
    return Boolean(TEMPLATE_DEFINITIONS[normalized]);
  }

  function listTemplates() {
    return Object.keys(TEMPLATE_DEFINITIONS).map((key) => {
      const item = TEMPLATE_DEFINITIONS[key];
      return {
        id: item.id,
        label: item.label,
        modality: item.modality,
        description: item.description,
        sectionOrder: item.sectionOrder.slice(),
        sectionLabels: Object.assign({}, item.sectionLabels)
      };
    });
  }

  function getTemplateOptions() {
    return listTemplates().map((item) => ({
      value: item.id,
      label: item.label,
      modality: item.modality,
      description: item.description
    }));
  }

  function getChecklist(templateId) {
    return getTemplate(templateId).checklist || [];
  }

  function buildEmptySectionState(templateId, includeDefaults = true) {
    const template = getTemplate(templateId);
    const state = {};
    template.sectionOrder.forEach((sectionKey) => {
      state[sectionKey] = includeDefaults
        ? String(template.sectionDefaults[sectionKey] || '')
        : '';
    });
    return state;
  }

  function buildSectionArray(templateId, includeDefaults = true) {
    const template = getTemplate(templateId);
    return template.sectionOrder.map((sectionKey) => ({
      key: sectionKey,
      label: template.sectionLabels[sectionKey] || titleCase(sectionKey),
      text: includeDefaults ? String(template.sectionDefaults[sectionKey] || '') : ''
    }));
  }

  function renderTemplateSkeleton(templateId, options) {
    const template = getTemplate(templateId);
    const includeDefaults = !(options && options.includeDefaults === false);
    const parts = [];
    parts.push('FINDINGS:');
    template.sectionOrder.forEach((sectionKey) => {
      const label = template.sectionLabels[sectionKey] || titleCase(sectionKey);
      const text = includeDefaults ? String(template.sectionDefaults[sectionKey] || '') : '';
      parts.push(`${label}: ${text}`.trim());
    });
    parts.push('');
    parts.push('IMPRESSION:');
    return parts.join('\n').trim();
  }

  function inferTemplateId(input) {
    const text = String(input || '').trim().toLowerCase();
    if (!text) return DEFAULT_TEMPLATE_ID;

    if (MODALITY_ALIASES[text]) {
      return MODALITY_ALIASES[text];
    }
    if (TEMPLATE_DEFINITIONS[text]) {
      return text;
    }

    if (text.includes('brain') || text.includes('intracranial') || text.includes('head mri')) {
      return 'mri-brain';
    }
    if (text.includes('abdomen') || text.includes('pelvis') || text.includes('abdominopelvic') || text.includes('a/p')) {
      return 'ct-abdomen-pelvis';
    }
    if (text.includes('xray') || text.includes('x-ray') || text.includes('radiograph') || text.includes('cxr')) {
      return 'xr-chest';
    }
    if (text.includes('chest') || text.includes('lung') || text.includes('thorax')) {
      return 'ct-chest';
    }
    return DEFAULT_TEMPLATE_ID;
  }

  function titleCase(value) {
    return String(value || '')
      .replace(/[_-]+/g, ' ')
      .split(' ')
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  function getDefaultTemplateId() {
    return DEFAULT_TEMPLATE_ID;
  }

  const api = {
    DEFAULT_TEMPLATE_ID,
    TEMPLATES: clone(TEMPLATE_DEFINITIONS),
    normalizeTemplateId,
    hasTemplate,
    getTemplate,
    listTemplates,
    getTemplateOptions,
    getChecklist,
    buildEmptySectionState,
    buildSectionArray,
    renderTemplateSkeleton,
    inferTemplateId,
    getDefaultTemplateId
  };

  return Object.freeze(api);
});
